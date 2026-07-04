"""
multi_agent_orchestrator.py — v6 (Order Zone Strategy, gate + PnL tracking fixes)
==================================================================================
Updated pipeline integrating the FTT strategy:

  Agent 1 — OrderZoneAgent     : 1H HH/LL box identification
  Agent 2 — SetupDetectionAgent: 5M breakout + EMA20 retracement
  Agent 3 — MicroConfirmAgent  : 5s/10s/15s/20s/30s/1m alignment
  Agent 4 — ExecutionAgent     : Places 1M or 5M FTT trade
  Agent 5 — RiskAgent          : SL/TP/drawdown management
  Agent 6 — LoggingAgent       : Timestamped IST logs per pair

v6 CHANGES (bug fixes only — no core strategy logic altered):
  FIX 1 — regime_hint is no longer forced to BULL_TREND/BEAR_TREND from a
          binary UPTREND/DOWNTREND read. strategy_engine.py v10 now reports
          a real trend ("UPTREND" | "DOWNTREND" | "RANGING"), so RANGING
          maps straight through to the AI pipeline's RANGING regime,
          letting the PPO hard-block fire correctly.
  FIX 2 — trend_strength from the TradeSignal (HH/HL confirmation strength)
          is now passed into the AI pipeline's market_context so the meta-
          classifier's structure score is independent of confidence.
  FIX 3 — _resolve_trades() now actually calls risk_agent.resolve_expired()
          with live prices, so daily_pnl and MAX_DAILY_DD are no longer
          dead code, and wins/losses get logged automatically instead of
          requiring a manual platform check.

Run:
    python multi_agent_orchestrator.py

Logs show exactly which pair, what signal, at what IST time — every cycle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(__file__))

from agents.strategy_engine import (
    FTTStrategyEngine,
    MultiPairScanner,
    TradeSignal,
    Signal,
    FOREX_PAIRS,
)
from agents.data_provider_otc import make_otc_provider   # OTC tick buffer
from agents.ai_agents import RevisedPipelineAgent

# ── Logging setup ─────────────────────────────────────────────────────────────

IST = pytz.timezone("Asia/Kolkata")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, "multi_agent_system.log"), encoding="utf-8"
        ),
    ],
)
log = logging.getLogger("orchestrator_v6")

# Live price cache shared between the orchestrator and the OTC data provider
_LIVE_PRICE_CACHE: Dict[str, float] = {}

# OTC pair names — these use the browser tick buffer instead of yfinance
_OTC_PAIRS = {"EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "EURGBP-OTC",
              "AUDCAD-OTC", "NZDUSD-OTC", "USDCHF-OTC", "EURCAD-OTC"}

# ── Optional imports (degrade gracefully) ─────────────────────────────────────

try:
    from config import settings
    TRADE_AMOUNT    = getattr(settings, "TRADE_AMOUNT", 1.0)
    TRADE_MODE      = getattr(settings, "TRADE_MODE", "ftt")
    CONFIDENCE_GATE = getattr(settings, "CONFIDENCE_THRESHOLD", 0.70)
    MODEL_DIR       = getattr(settings, "MODEL_DIR", "models")
    JOURNAL_PATH    = getattr(settings, "JOURNAL_PATH", os.path.join(LOG_DIR, "journal.csv"))
except Exception:
    TRADE_AMOUNT    = 1.0
    TRADE_MODE      = "ftt"
    CONFIDENCE_GATE = 0.70
    MODEL_DIR       = "models"
    JOURNAL_PATH    = os.path.join(LOG_DIR, "journal.csv")

try:
    from bot.browser import OlymtradeBot
    BOT_AVAILABLE = True
except Exception:
    BOT_AVAILABLE = False
    log.warning("[Init] OlymtradeBot not available — paper-trade mode")

try:
    from bot.risk import RiskManager, TradeRecord
    RISK_AVAILABLE = True
except Exception:
    RISK_AVAILABLE = False
    log.warning("[Init] RiskManager not available — skipping journal")

try:
    from data.fetcher import fetch_candles
    DATA_AVAILABLE = True
except Exception:
    DATA_AVAILABLE = False
    log.warning("[Init] data.fetcher not available — using yfinance fallback")


# ── Data provider (real-market pairs via yfinance) ────────────────────────────

def _yfinance_tf_map(tf_label: str) -> str:
    return {
        "1h":  "1h",
        "5m":  "5m",
        "1m":  "1m",
        "5s":  "1m",
        "10s": "1m",
        "15s": "1m",
        "20s": "1m",
        "30s": "1m",
    }.get(tf_label, "1m")


def _resample_to_seconds(df_1m, seconds: int):
    import pandas as pd
    if df_1m is None or df_1m.empty:
        return df_1m
    rule = f"{seconds}s"
    try:
        df = df_1m.resample(rule).agg({
            "open":  "first",
            "high":  "max",
            "low":   "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return df
    except Exception:
        return df_1m


def _is_stale_frame(df: pd.DataFrame, tf_label: str) -> bool:
    if df is None or df.empty or len(df) < 3:
        return True
    if "close" not in df.columns or "open" not in df.columns:
        return True
    try:
        last_idx = pd.Timestamp(df.index[-1])
        if last_idx.tzinfo is None:
            last_idx = last_idx.tz_localize("UTC")
        last_idx = last_idx.tz_convert(IST)
        age_min = (datetime.now(IST) - last_idx).total_seconds() / 60.0
        if age_min > 90:
            return True
        close = pd.to_numeric(df["close"], errors="coerce")
        if close.tail(3).nunique() <= 1:
            return True
    except Exception:
        return True
    return False


# Per-pair plausible price ranges. JPY-quoted pairs trade ~100-200, everything
# else in _OTC_PAIRS trades ~0.5-2.0 — a single hardcoded [0.5, 5.0] range
# (as used previously here and in data_provider_otc.py, and in the older
# tg_algo_adv AUD/USD bug) silently rejects every valid USDJPY-OTC price.
_PAIR_PRICE_RANGES = {
    "USDJPY-OTC": (50.0, 250.0),
}
_DEFAULT_PRICE_RANGE = (0.01, 5.0)


def _is_plausible_price(pair: str, live_price: float) -> bool:
    try:
        p = float(live_price)
    except Exception:
        return False
    lo, hi = _PAIR_PRICE_RANGES.get(pair, _DEFAULT_PRICE_RANGE)
    return lo <= p <= hi


def _inject_live_price(df: pd.DataFrame, live_price: float, tf_label: str, pair: str = "") -> pd.DataFrame:
    if df is None or df.empty or live_price is None or not _is_plausible_price(pair, live_price):
        return df
    try:
        out = df.copy()
        out = out.sort_index()
        ts = pd.Timestamp(datetime.now(IST))
        if tf_label in ("1m", "m1", "M1"):
            ts = ts.floor("1min")
        elif tf_label in ("5m", "m5", "M5"):
            ts = ts.floor("5min")
        else:
            ts = ts.floor("1h")
        if len(out) > 0 and ts <= pd.Timestamp(out.index[-1]):
            idx = out.index[-1]
            out.loc[idx, ["open", "high", "low", "close"]] = live_price
        else:
            out.loc[ts, ["open", "high", "low", "close"]] = live_price
            out.loc[ts, "volume"] = 0
        out = out.tail(200)
        return out
    except Exception:
        return df


def build_data_provider(pair: str):
    """Returns a callable (pair, tf_label) -> pd.DataFrame for real-market pairs."""
    def provider(p: str, tf_label: str) -> pd.DataFrame:
        yf_interval = _yfinance_tf_map(tf_label)

        if DATA_AVAILABLE:
            try:
                df = fetch_candles(p, yf_interval.upper().replace("H", "H"), n=200)
                if not df.empty:
                    if tf_label in ("5s", "10s", "15s", "20s", "30s"):
                        secs = int(tf_label.replace("s", ""))
                        df = _resample_to_seconds(df, secs)
                    df.columns = [c.lower() for c in df.columns]
                    live_price = _LIVE_PRICE_CACHE.get(p)
                    if live_price is not None and (
                        tf_label in ("1m", "5m", "m1", "m5", "M1", "M5")
                        or _is_stale_frame(df, tf_label)
                    ):
                        df = _inject_live_price(df, live_price, tf_label, pair=p)
                    return df
            except Exception as e:
                log.debug(f"[DataProvider] fetch_candles failed for {p}/{tf_label}: {e}")

        try:
            import yfinance as yf
            period = "5d" if yf_interval in ("1m", "5m") else "60d"
            ticker = yf.Ticker(f"{p}=X")
            df = ticker.history(period=period, interval=yf_interval)
            if df.empty:
                return pd.DataFrame()
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
            if tf_label in ("5s", "10s", "15s", "20s", "30s"):
                secs = int(tf_label.replace("s", ""))
                df = _resample_to_seconds(df, secs)
            live_price = _LIVE_PRICE_CACHE.get(p)
            if live_price is not None and (
                tf_label in ("1m", "5m", "m1", "m5", "M1", "M5")
                or _is_stale_frame(df, tf_label)
            ):
                df = _inject_live_price(df, live_price, tf_label, pair=p)
            return df
        except Exception as e:
            log.warning(f"[DataProvider] yfinance fallback failed {p}/{tf_label}: {e}")
            return pd.DataFrame()

    return provider


# ── Logging Agent ─────────────────────────────────────────────────────────────

class LoggingAgent:
    TRADE_LOG = os.path.join(LOG_DIR, "trade_signals.log")

    def log_signal(self, t: TradeSignal) -> None:
        line = (
            f"{t.ist_time} | PAIR={t.pair} | TF={t.trade_duration_min}M "
            f"| TREND={t.trend} | SIGNAL={t.signal.value} "
            f"| ENTRY={t.entry_price:.5f} | SL={t.sl:.5f} | TP={t.tp:.5f} "
            f"| ZONE={t.order_zone.lower:.5f}–{t.order_zone.upper:.5f} "
            f"| CONF={t.confidence:.0%} | {t.reason}"
        )
        log.info(f"[LogAgent] {line}")
        with open(self.TRADE_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_scan_start(self, pairs: List[str]) -> None:
        ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        log.info(f"[LogAgent] {ts} | Scanning {len(pairs)} pairs: {', '.join(pairs)}")

    def log_no_setup(self, ts: str) -> None:
        log.info(f"[LogAgent] {ts} | No confirmed setup this cycle — waiting.")

    def log_result(self, pair: str, signal: str, result: str, pnl: float) -> None:
        ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        log.info(f"[LogAgent] {ts} | RESULT | {pair} {signal} → {result} PnL=${pnl:+.2f}")
        with open(self.TRADE_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts} | RESULT | {pair} {signal} | {result} | PnL={pnl:+.2f}\n")


# ── Risk Agent ────────────────────────────────────────────────────────────────

class RiskAgent:
    MAX_OPEN     = 3
    MAX_DAILY_DD = 0.05
    MAX_RISK_PCT = 0.02

    def __init__(self):
        self.open_trades: List[Dict] = []
        self.daily_pnl   = 0.0

    def can_trade(self, balance: float = 100.0):
        if len(self.open_trades) >= self.MAX_OPEN:
            return False, f"Max open trades ({self.MAX_OPEN}) reached"
        if self.daily_pnl <= -balance * self.MAX_DAILY_DD:
            return False, f"Daily drawdown limit hit (${self.daily_pnl:.2f})"
        return True, "OK"

    def register_open(self, trade: TradeSignal, amount: float):
        self.open_trades.append({
            "pair":      trade.pair,
            "signal":    trade.signal.value,
            "entry":     trade.entry_price,
            "sl":        trade.sl,
            "tp":        trade.tp,
            "amount":    amount,
            "placed_at": datetime.now(IST),
            "duration":  trade.trade_duration_min * 60,
        })

    def resolve_expired(self, current_prices: Dict[str, float]) -> List[Dict]:
        now = datetime.now(IST)
        resolved, remaining = [], []
        for t in self.open_trades:
            age = (now - t["placed_at"]).total_seconds()
            if age < t["duration"]:
                remaining.append(t)
                continue
            price  = current_prices.get(t["pair"], t["entry"])
            win    = (price > t["entry"]) if t["signal"] == "BUY" else (price < t["entry"])
            result = "WIN" if win else "LOSS"
            pnl    = t["amount"] * 0.82 if win else -t["amount"]
            self.daily_pnl += pnl
            t.update({"result": result, "pnl": pnl, "exit_price": price})
            resolved.append(t)
        self.open_trades = remaining
        return resolved


# ── Execution Agent ───────────────────────────────────────────────────────────

class ExecutionAgent:
    def __init__(self, bot=None):
        self.bot = bot

    async def place(self, trade: TradeSignal, amount: float) -> bool:
        direction = "call" if trade.signal == Signal.BUY else "put"
        if self.bot and BOT_AVAILABLE:
            try:
                ok = await self.bot.place_ftt_trade(
                    signal=trade.signal,
                    duration_seconds=trade.trade_duration_min * 60,
                    asset=trade.pair,
                )
                return ok
            except Exception as e:
                log.error(f"[ExecAgent] Bot trade failed: {e}")
                return False
        else:
            log.info(
                f"[ExecAgent] 📝 PAPER TRADE | {trade.pair} {direction.upper()} "
                f"{trade.trade_duration_min}M ${amount:.2f}"
            )
            return True


# ── Main Orchestrator ─────────────────────────────────────────────────────────

class MultiAgentTradingSystem:
    SCAN_INTERVAL    = 30.0
    RESOLVE_INTERVAL = 5.0

    def __init__(self, pairs: List[str] = FOREX_PAIRS, trade_duration_min: int = 1):
        self.pairs              = pairs
        self.trade_duration_min = trade_duration_min
        self.scanner            = MultiPairScanner(pairs)
        self.risk_agent         = RiskAgent()
        self.logger_agent       = LoggingAgent()
        self.exec_agent         = ExecutionAgent()
        self.pipeline_agent     = RevisedPipelineAgent("EURUSD-OTC")
        self.running            = False
        self._last_scan         = 0.0
        self._last_resolve      = 0.0
        self._total             = 0
        self._pair_cooldown: Dict[str, float] = {}
        # Cached OTC data provider — created once after bot is ready
        self._otc_dp            = None

    async def setup(self) -> bool:
        log.info("═" * 70)
        log.info("  FTT Multi-Agent System v6 | Order Zone Strategy")
        log.info(f"  Pairs: {', '.join(self.pairs)}")
        log.info(f"  Trade Duration: {self.trade_duration_min}M")
        log.info(f"  Confidence Gate: {CONFIDENCE_GATE:.0%}")
        log.info(f"  Scan interval: {self.SCAN_INTERVAL}s")
        log.info("═" * 70)

        if BOT_AVAILABLE:
            bot = OlymtradeBot()
            await bot.start()
            if not await bot.login():
                log.error("Bot login failed — running in paper mode.")
            else:
                self.exec_agent.bot = bot
                log.info("[Setup] Bot login OK")

                # ── OTC data provider using the live browser page ──
                has_otc = any(p in _OTC_PAIRS for p in self.pairs)
                if has_otc and bot._page is not None:
                    self._otc_dp = make_otc_provider(bot._page, _LIVE_PRICE_CACHE)
                    log.info("[Setup] OTC browser tick-buffer data provider ready")

                # Pre-select the asset in the browser so live prices start flowing
                for pair in self.pairs:
                    try:
                        selected = await bot.select_asset(pair)
                        if not selected:
                            log.warning(f"[Setup] Asset selection failed for {pair}")
                        await asyncio.sleep(1.0)
                    except Exception as e:
                        log.warning(f"[Setup] Asset selection error for {pair}: {e}")

        return True

    async def run(self):
        if not await self.setup():
            return
        self.running = True
        try:
            while self.running:
                now = time.monotonic()
                if now - self._last_resolve >= self.RESOLVE_INTERVAL:
                    self._last_resolve = now
                    await self._resolve_trades()
                if now - self._last_scan >= self.SCAN_INTERVAL:
                    self._last_scan = now
                    await self._scan_and_trade()
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            log.info("Shutdown requested.")
        except Exception as e:
            log.critical(f"Fatal: {e}", exc_info=True)
        finally:
            await self._shutdown()

    async def _scan_and_trade(self):
        ts_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        self.logger_agent.log_scan_start(self.pairs)

        ok, reason = self.risk_agent.can_trade()
        if not ok:
            log.info(f"[Orchestrator] Risk gate blocked: {reason}")
            return

        # ── Refresh live price for ONE pair per cycle (round-robin) ──
        # BUG FIX: get_current_price() reads whatever chart is currently open in
        # the browser — it does NOT take a pair argument. The old code called it
        # once per pair without switching charts in between, so every pair in
        # self.pairs was overwritten with the SAME price (whichever asset
        # happened to be visually open). Full select_asset() per pair per cycle
        # is too slow for a 30s scan cycle (each call does UI menu clicks + sleeps),
        # so instead we rotate: switch to exactly one pair per scan cycle, read
        # its price, advance the pointer. Over N cycles all N pairs get refreshed.
        if self.exec_agent.bot and BOT_AVAILABLE and self.pairs:
            idx = getattr(self, "_price_rotation_idx", 0) % len(self.pairs)
            pair = self.pairs[idx]
            self._price_rotation_idx = idx + 1
            try:
                selected = await self.exec_agent.bot.select_asset(pair)
                if not selected:
                    log.warning(f"[Orchestrator] Could not switch to {pair} for price refresh")
                else:
                    live_price = await self.exec_agent.bot.get_current_price()
                    if live_price is not None and _is_plausible_price(pair, live_price):
                        _LIVE_PRICE_CACHE[pair] = live_price
                        log.info(f"[Orchestrator] Live price {pair}: {live_price:.5f}")
                    else:
                        log.warning(
                            f"[Orchestrator] {pair}: implausible/missing price "
                            f"({live_price!r}) — check CSS selectors in get_current_price()"
                        )
            except Exception as e:
                log.warning(f"[Orchestrator] Price refresh failed for {pair}: {e}", exc_info=True)

        # ── Build data provider ────────────────────────────────────────────
        # OTC pairs → browser tick buffer   (live price from page)
        # Real pairs → yfinance / fetcher   (unchanged)
        otc_dp = self._otc_dp

        def data_provider(pair: str, tf_label: str) -> pd.DataFrame:
            if pair in _OTC_PAIRS and otc_dp is not None:
                return otc_dp(pair, tf_label)
            return build_data_provider(pair)(pair, tf_label)

        best = self.scanner.scan(data_provider, self.trade_duration_min)

        if best is None:
            self.logger_agent.log_no_setup(ts_ist)
            return

        # v6 FIX: regime_hint now reflects RANGING when the strategy engine
        # reports a real RANGING trend, instead of collapsing every non-
        # DOWNTREND signal into BULL_TREND. This lets the PPO's RANGING
        # hard-block actually fire when it should.
        if best.trend == "UPTREND":
            regime_hint = "BULL_TREND"
        elif best.trend == "DOWNTREND":
            regime_hint = "BEAR_TREND"
        else:
            regime_hint = "RANGING"

        pipeline_out = self.pipeline_agent.analyze_setup(
            best.signal.value,
            float(best.confidence),
            {
                "regime": regime_hint,
                "trend_strength": float(getattr(best, "trend_strength", 0.0)),  # v6: real structure score
                "impulse": 0.65,
                "base": 0.60,
                "zone": 0.65,
                "volatility": 0.10,
            },
        )

        best.confidence = float(pipeline_out["confidence"])
        best.reason = (
            f"{best.reason} | AI pipeline: {pipeline_out['regime']} "
            f"action={pipeline_out['action']} win={pipeline_out['win_prob']:.0%}"
        )

        if pipeline_out["action"] == "WAIT":
            log.info(
                f"[Orchestrator] AI pipeline held trade | {best.pair} {best.signal.value} "
                f"| {best.reason}"
            )
            self.logger_agent.log_no_setup(ts_ist)
            return

        if best.confidence < CONFIDENCE_GATE:
            self.logger_agent.log_no_setup(ts_ist)
            return

        self.logger_agent.log_signal(best)

        success = await self.exec_agent.place(best, TRADE_AMOUNT)
        if success:
            self.risk_agent.register_open(best, TRADE_AMOUNT)
            self._total += 1
            log.info(
                f"[Orchestrator] ✅ Trade placed | {best.pair} {best.signal.value} "
                f"| {best.trade_duration_min}M | entry={best.entry_price:.5f} "
                f"| sl={best.sl:.5f} | tp={best.tp:.5f}"
            )
        else:
            log.warning(f"[Orchestrator] ❌ Trade placement failed for {best.pair}")

    async def _resolve_trades(self):
        # v6 FIX: this used to only log "expired — check platform" and drop
        # the trade without ever calling risk_agent.resolve_expired(), so
        # daily_pnl and MAX_DAILY_DD never updated. It now resolves against
        # the latest known live price for each pair and logs WIN/LOSS + PnL.
        if not self.risk_agent.open_trades:
            return

        current_prices: Dict[str, float] = {}
        if self.exec_agent.bot and BOT_AVAILABLE:
            for t in self.risk_agent.open_trades:
                pair = t["pair"]
                if pair in current_prices:
                    continue
                cached = _LIVE_PRICE_CACHE.get(pair)
                if cached is not None:
                    current_prices[pair] = cached

        resolved = self.risk_agent.resolve_expired(current_prices)
        for t in resolved:
            self.logger_agent.log_result(t["pair"], t["signal"], t["result"], t["pnl"])
            log.info(
                f"[Orchestrator] 🔔 Trade resolved | {t['pair']} {t['signal']} "
                f"→ {t['result']} | entry={t['entry']:.5f} exit={t.get('exit_price', t['entry']):.5f} "
                f"| PnL=${t['pnl']:+.2f} | daily_pnl=${self.risk_agent.daily_pnl:+.2f}"
            )

    async def _shutdown(self):
        self.running = False
        log.info("═" * 70)
        log.info(f"  Session closed | trades placed={self._total} | daily_pnl=${self.risk_agent.daily_pnl:+.2f}")
        log.info(f"  ⚠️  Verify actual win/loss on the Olymptrade platform as well.")
        log.info("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FTT Multi-Agent Orchestrator v6")
    parser.add_argument("--duration", type=int, default=1, choices=[1, 5],
                        help="FTT trade duration in minutes (1 or 5)")
    parser.add_argument("--pairs", nargs="+", default=FOREX_PAIRS,
                        help="Forex pairs to scan")
    args = parser.parse_args()

    system = MultiAgentTradingSystem(
        pairs              = args.pairs,
        trade_duration_min = args.duration,
    )
    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        pass
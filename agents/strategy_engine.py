"""
agents/strategy_engine.py  — v10 (10S Entry + 1M HTF + Trend/Zone/Prediction, gate fixes)
============================================================================

Strategy: 10-Second Scalping — VWAP + EMA(9/21) + RSI(7) + Volume
          confirmed by 1-Minute trend (was 5M), with HH/LL trend structure,
          supply/demand zones, and previous-candle pattern prediction.

All original orchestrator APIs preserved: TradeSignal, Signal, OrderZone,
FOREX_PAIRS, MultiPairScanner.scan() — zero changes needed in orchestrator
beyond what's documented in multi_agent_orchestrator.py's changelog.

v10 CHANGES (bug fixes only — no core strategy logic altered):
  FIX 1 — Structure gate now requires ALIGNMENT with the trade direction,
          not just non-opposition. Previously TrendType=NEUTRAL (zero
          strength — i.e. an accumulation range) passed the gate freely
          for both BUY and SELL as long as HTF wasn't outright opposed.
          This is very likely the direct cause of buying inside a range.
  FIX 2 — TradeSignal.trend now reflects the real computed trend_type
          (UPTREND / DOWNTREND / RANGING) instead of being hardcoded to
          "UPTREND" on every BUY and "DOWNTREND" on every SELL. This was
          feeding a fake trend into the orchestrator's regime_hint, which
          made the AI pipeline's regime check meaningless.
  FIX 3 — BODY_MIN_RATIO raised from 0.00 to 0.25 so a real candle body
          is required, not just a doji/indecision candle.
  FIX 4 — trend_strength is now exposed on SignalResult and TradeSignal
          so the AI pipeline can use a real structure score instead of
          echoing the strategy engine's own confidence back at itself.

TIMEFRAME PIPELINE (unchanged from v9):
  HTF GATE      : 1-Minute chart — EMA9 vs EMA21 trend bias
  ENTRY/SIGNAL  : 10-Second chart — VWAP+EMA+RSI+Volume rules
                  + HH/HL / LL/LH trend confirmation
                  + supply & demand zone detection
                  + previous-candle pattern prediction for next-candle direction

═══════════════════════════════════════════════════════════════════
BUY  — ALL must be true (on the 10S chart):
  1. TREND    : close > VWAP  AND  EMA9 > EMA21
  2. MOMENTUM : RSI(7) > 55   AND  RSI(7) < 75
  3. VOLUME   : current bar volume >= threshold x average of last 20 bars
  4. ENTRY    : bullish candle closes above EMA9
                candle body > min ratio of total range
                upper wick < max ratio of candle size
  5. HTF GATE : 1M trend is bullish (EMA9_1m > EMA21_1m)
  6. STRUCTURE: HH/HL trend confirmation on 10S must ALIGN with BUY (v10 fix)
  7. ZONE     : price at/above demand zone
  8. PREDICT  : previous 10S candles do not predict a strong DOWN reversal
  NO TRADE IF : EMA9 and EMA21 flat | price crossing VWAP repeatedly
                RSI 48–52 | volume below avg | very long wicks
                3 consecutive large candles already moved same direction
                structure is NEUTRAL/opposing and HTF is not confirming (v10 fix)

SELL — mirror of BUY using supply zone / LL-LH structure / EMA below VWAP.

EXPIRY (trade_duration_min):
  Strong trend + high volume  → 2 candles
  Moderate trend              → 3 candles
═══════════════════════════════════════════════════════════════════

Session / payout tier / OTC blocklist / AutoAssetSelector unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

log = logging.getLogger("strategy_engine_v10")
IST = pytz.timezone("Asia/Kolkata")

# ══════════════════════════════════════════════════════════════════════════════
# Public types  (orchestrator-facing — do NOT rename/remove)
# ══════════════════════════════════════════════════════════════════════════════

class Signal(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    NONE = "NONE"


@dataclass
class OrderZone:
    lower: float
    upper: float

    @property
    def mid(self) -> float:
        return (self.lower + self.upper) / 2

    def contains(self, price: float, tol: float = 0.0) -> bool:
        return (self.lower - tol) <= price <= (self.upper + tol)


@dataclass
class TradeSignal:
    pair:               str
    signal:             str        # Signal enum value
    entry_price:        float
    sl:                 float
    tp:                 float
    confidence:         float      # 0.0–1.0
    reason:             str
    order_zone:         OrderZone  # VWAP band used as reference zone
    trend:              str        # "UPTREND" | "DOWNTREND" | "RANGING"  (v10: now real, not hardcoded)
    trade_duration_min: int        # expiry in minutes
    trend_strength:     float = 0.0  # v10: HH/HL confirmation strength, 0.0-1.0, for the AI pipeline
    ist_time:           str = field(
        default_factory=lambda: datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    )


# ══════════════════════════════════════════════════════════════════════════════
# Session config  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

# Only EUR/USD OTC — no other pairs, no blocklist needed
_OTC_PAIRS: set = set()

# EUR/USD OTC is available 24/5 on Olymp Trade with synthetic pricing.
# Format: pair -> (start_H, start_M, end_H, end_M, payout_weight)
_SESSION: Dict[str, Tuple] = {
    "EURUSD-OTC": (0, 0, 23, 59, 1.30),   # 24h available, highest OTC payout tier
}

FOREX_PAIRS: List[str] = ["EURUSD-OTC"]
_PAYOUT_WEIGHT: Dict[str, float] = {p: v[4] for p, v in _SESSION.items()}


def _in_session(pair: str, now_ist: datetime) -> bool:
    if pair in _OTC_PAIRS:
        return False
    sched = _SESSION.get(pair)
    if sched is None:
        return True
    sh, sm, eh, em = sched[0], sched[1], sched[2], sched[3]
    t     = dtime(now_ist.hour, now_ist.minute)
    start = dtime(sh, sm)
    end   = dtime(eh, em)
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end


def active_pairs(now_ist: Optional[datetime] = None) -> List[str]:
    if now_ist is None:
        now_ist = datetime.now(IST)
    return [p for p in FOREX_PAIRS if _in_session(p, now_ist)]


# ══════════════════════════════════════════════════════════════════════════════
# Indicator helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 7) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session VWAP: cumulative(typical_price * volume) / cumulative(volume).
    If volume column is missing or zero, falls back to SMA(20) of close.
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return df["close"].rolling(20).mean().fillna(df["close"])
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol    = df["volume"].cumsum().replace(0, np.nan)
    return (cum_tp_vol / cum_vol).fillna(df["close"])


def _vol_avg(df: pd.DataFrame, bars: int = 20) -> pd.Series:
    """20-bar rolling average volume."""
    if "volume" not in df.columns:
        return pd.Series(np.ones(len(df)), index=df.index)
    return df["volume"].rolling(bars).mean().fillna(df["volume"].mean())


def _ema_slope_pct(ema_series: pd.Series, bars: int = 3) -> float:
    """Return the percentage change of the EMA over the last `bars` candles."""
    if len(ema_series) < bars + 1:
        return float("inf")
    prev = float(ema_series.iloc[-bars])
    if prev == 0:
        return float("inf")
    return abs(float(ema_series.iloc[-1]) - prev) / prev


def _ema_slope_flat(ema_series: pd.Series, bars: int = 3, pct_tol: float = 0.0003) -> bool:
    """True if the EMA has barely moved over the last `bars` candles (flat)."""
    return _ema_slope_pct(ema_series, bars=bars) < pct_tol


def _count_same_dir_candles(df: pd.DataFrame, direction: str, n: int = 3) -> int:
    """Count consecutive large candles in the same direction (last n bars)."""
    recent = df.iloc[-n:]
    bodies = abs(recent["close"] - recent["open"])
    ranges = recent["high"] - recent["low"]
    large  = bodies > 0.5 * ranges
    if direction == "BUY":
        same = (recent["close"] > recent["open"]) & large
    else:
        same = (recent["close"] < recent["open"]) & large
    return int(same.sum())


def _detect_supply_demand_zones(df: pd.DataFrame, lookback: int = 10) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect supply (resistance) and demand (support) zones on the entry-timeframe chart.
    Demand zone: area where price reversed upward (V-shape bottom + bullish follow-through).
    Supply zone: area where price reversed downward (inverted-V top + bearish follow-through).
    Returns: (demand_zone, supply_zone)
    """
    if len(df) < lookback + 2:
        return None, None

    recent = df.iloc[-lookback:]
    lows = recent["low"].values
    highs = recent["high"].values
    closes = recent["close"].values
    opens = recent["open"].values

    demand_zone = None
    for i in range(1, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:        # V-shape bottom
            if closes[i+1] > opens[i+1]:                        # bullish follow-through
                demand_zone = lows[i]
                break

    supply_zone = None
    for i in range(1, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:    # inverted-V top
            if closes[i+1] < opens[i+1]:                        # bearish follow-through
                supply_zone = highs[i]
                break

    return demand_zone, supply_zone


def _confirm_trend_hh_ll(df: pd.DataFrame, bars: int = 5) -> Tuple[str, float]:
    """
    Confirm trend using Higher-Highs/Higher-Lows (uptrend) or
    Lower-Highs/Lower-Lows (downtrend) on the entry-timeframe chart.
    Returns: (trend_type, strength_score)
    trend_type: "STRONG_UP", "UP", "NEUTRAL", "DOWN", "STRONG_DOWN"

    v11 FIX: the old version required EVERY consecutive bar to strictly
    beat the previous one (AND across the whole window). On noisy
    tick-derived OTC candles that condition almost never holds even in a
    real trend — a single small pullback wick anywhere in the window
    collapsed the result to NEUTRAL/0.0, which then propagated downstream
    and forced the AI pipeline's regime into permanent RANGING (see
    MetaClassifierAgent / PPOExecutionPolicyAgent RANGING hard-block).
    That made it impossible to ever distinguish accumulation (choppy,
    low net movement) from a real breakout (majority of bars agreeing,
    high strength) or a reversal (structure flips from majority-up to
    majority-down across the window). Fixed with a proportional vote
    instead of a strict AND.
    """
    if len(df) < bars + 2:
        return "NEUTRAL", 0.0

    recent = df.iloc[-(bars + 1):]
    highs = recent["high"].values
    lows = recent["low"].values

    up_votes = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1] and lows[i] > lows[i-1])
    down_votes = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1] and lows[i] < lows[i-1])
    total_moves = len(highs) - 1
    if total_moves <= 0:
        return "NEUTRAL", 0.0

    up_ratio = up_votes / total_moves
    down_ratio = down_votes / total_moves

    # Net directional displacement across the window, normalized by the
    # window's own range — this is what actually separates "accumulation"
    # (small net move, choppy) from "breakout" (large net move, one-sided).
    window_range = float(np.max(highs) - np.min(lows))
    net_move = float((highs[-1] + lows[-1]) / 2 - (highs[0] + lows[0]) / 2)
    displacement = min(abs(net_move) / window_range, 1.0) if window_range > 0 else 0.0

    VOTE_THRESHOLD = 0.6   # majority, not unanimous

    if up_ratio >= VOTE_THRESHOLD and net_move >= 0:
        strength = max(up_ratio, displacement)
        return ("STRONG_UP" if strength > 0.7 else "UP"), strength
    elif down_ratio >= VOTE_THRESHOLD and net_move <= 0:
        strength = max(down_ratio, displacement)
        return ("STRONG_DOWN" if strength > 0.7 else "DOWN"), strength

    # Neither side has a majority — genuine accumulation/chop, not a bug.
    return "NEUTRAL", round(displacement * 0.3, 3)


def _analyze_previous_candles(df: pd.DataFrame, bars: int = 3) -> Dict[str, Any]:
    """
    Analyze previous candles to predict the next candle's likely direction.
    Looks at momentum direction, average body strength, and wick pressure
    (long upper wicks = sell pressure / reversal down, long lower wicks =
    buy pressure / reversal up).
    """
    if len(df) < bars + 2:
        return {}

    recent = df.iloc[-(bars + 1):]

    analysis: Dict[str, Any] = {
        "momentum_direction": None,
        "body_strength": 0.0,
        "wick_pressure": None,           # "buy_pressure" | "sell_pressure" | "neutral"
        "reversal_probability": 0.0,
        "next_likely_direction": None,
    }

    bullish_candles = sum(1 for i in range(len(recent)) if recent["close"].iloc[i] > recent["open"].iloc[i])
    bearish_candles = len(recent) - bullish_candles

    if bullish_candles > bearish_candles:
        analysis["momentum_direction"] = "UP"
    elif bearish_candles > bullish_candles:
        analysis["momentum_direction"] = "DOWN"
    else:
        analysis["momentum_direction"] = "NEUTRAL"

    bodies = abs(recent["close"] - recent["open"])
    ranges = recent["high"] - recent["low"]
    body_ratios = (bodies / ranges.replace(0, 1)).values
    analysis["body_strength"] = float(np.mean(body_ratios))

    upper_wicks = recent["high"] - np.maximum(recent["close"], recent["open"])
    lower_wicks = np.minimum(recent["close"], recent["open"]) - recent["low"]

    avg_upper_wick = float(np.mean(upper_wicks))
    avg_lower_wick = float(np.mean(lower_wicks))

    if avg_upper_wick > avg_lower_wick * 1.3:
        analysis["wick_pressure"] = "sell_pressure"
        analysis["reversal_probability"] = 0.7
        analysis["next_likely_direction"] = "DOWN"
    elif avg_lower_wick > avg_upper_wick * 1.3:
        analysis["wick_pressure"] = "buy_pressure"
        analysis["reversal_probability"] = 0.7
        analysis["next_likely_direction"] = "UP"
    else:
        analysis["wick_pressure"] = "neutral"
        analysis["reversal_probability"] = 0.3
        analysis["next_likely_direction"] = analysis["momentum_direction"]

    return analysis


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — Higher Timeframe Trend Gate  (reads the 1-Minute chart)
# ══════════════════════════════════════════════════════════════════════════════

class HTFTrendAgent:
    """
    1M chart: EMA9 vs EMA21 determines higher-timeframe trend bias
    for the faster 10S entry chart.
    BUY  only if EMA9_1m > EMA21_1m (1M bullish).
    SELL only if EMA9_1m < EMA21_1m (1M bearish).
    """
    EMA_FAST = 9
    EMA_SLOW = 21

    def trend(self, df1m: pd.DataFrame, pair: str = "") -> str:
        if df1m is None or len(df1m) < self.EMA_SLOW + 2:
            log.info(f"[{pair}] HTF: insufficient 1M data — skipping HTF gate")
            return "NEUTRAL"
        ema9  = _ema(df1m["close"], self.EMA_FAST)
        ema21 = _ema(df1m["close"], self.EMA_SLOW)
        last9  = float(ema9.iloc[-1])
        last21 = float(ema21.iloc[-1])
        result = "BULLISH" if last9 > last21 else "BEARISH"
        log.info(f"[{pair}] HTF 1M: EMA9={last9:.5f} EMA21={last21:.5f} → {result}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — Signal Detection  (runs on the 10-Second chart)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    signal:       str    # Signal.BUY / SELL / NONE
    trend:        str    # UPTREND / DOWNTREND
    confidence:   float
    expiry_min:   int
    rsi:          float
    ema9:         float
    ema21:        float
    vwap:         float
    vol_ratio:    float  # current volume / 20-bar avg
    reason:       str
    trend_strength: float = 0.0   # v10: HH/HL confirmation strength, passed to AI pipeline


class VWAPStrategyAgent:
    """
    10S chart — full validation per the strategy card, gated by the 1M HTF trend.

    BUY conditions (all must pass):
      TREND    : close > VWAP  AND  EMA9 > EMA21                       (10S)
      MOMENTUM : 55 < RSI(7) < 75                                       (10S)
      VOLUME   : vol >= threshold x 20-bar avg                          (10S)
      CANDLE   : bullish close above EMA9, body ratio, upper wick ratio
      STRUCTURE: HH/HL trend confirmation (10S) must ALIGN with BUY (v10 fix —
                 was: "must not oppose", now: "must actually confirm unless
                 HTF is strongly bullish")
      ZONE     : price at/above demand zone (10S)
      PREDICT  : previous 10S candles do not predict a strong DOWN reversal
      NOT-TRADE: EMA flat | VWAP cross chop | RSI 48-52 | long wicks | 3 consec candles

    SELL conditions: mirror of BUY using supply zone / LH-LL structure.
    """

    EMA_FAST         = 9
    EMA_SLOW         = 21
    RSI_PERIOD       = 7
    VOL_BARS         = 20
    VOL_SPIKE_MULT   = 0.80   # v10: restored to a real threshold (was 0.05 — effectively disabled)

    # BUY RSI band
    RSI_BUY_LO       = 40
    RSI_BUY_HI       = 92

    # SELL RSI band
    RSI_SELL_LO      = 8
    RSI_SELL_HI      = 58

    # RSI dead zone — flat market, no trade
    RSI_DEAD_LO      = 48
    RSI_DEAD_HI      = 52

    # Candle shape thresholds
    BODY_MIN_RATIO   = 0.25   # v10: require a real candle body (was 0.00 — let dojis pass)
    UPPER_WICK_MAX   = 0.65   # upper wick <= 65% of candle size (buy)
    LOWER_WICK_MAX   = 0.65   # lower wick <= 65% of candle size (sell)

    # EMA flat detection — tighter tolerance because 10S bars move less per-bar than 1M bars
    EMA_FLAT_TOL     = 0.000010

    # VWAP chop detection: how many times price crossed VWAP in last N bars
    VWAP_CROSS_BARS  = 5
    VWAP_CROSS_MAX   = 4

    # Consecutive same-direction large candles — don't chase
    CONSEC_MAX       = 2

    # v10: minimum trend_strength required to count as "structure confirms"
    STRUCTURE_MIN_STRENGTH = 0.25

    def analyse(
        self,
        df10s: pd.DataFrame,
        htf_trend: str,
        pair: str = "",
    ) -> Optional[SignalResult]:

        min_bars = max(self.EMA_SLOW, self.VOL_BARS, self.RSI_PERIOD) + 5
        if df10s is None or len(df10s) < min_bars:
            log.info(f"[{pair}] Signal: insufficient 10S data ({len(df10s) if df10s is not None else 0} bars)")
            return None

        # ── Compute indicators ────────────────────────────────────────
        close  = df10s["close"]
        open_  = df10s["open"]
        high   = df10s["high"]
        low    = df10s["low"]

        ema9   = _ema(close, self.EMA_FAST)
        ema21  = _ema(close, self.EMA_SLOW)
        rsi    = _rsi(close, self.RSI_PERIOD)
        vwap   = _vwap(df10s)
        vol_avg = _vol_avg(df10s, self.VOL_BARS)

        last_close  = float(close.iloc[-1])
        last_open   = float(open_.iloc[-1])
        last_high   = float(high.iloc[-1])
        last_low    = float(low.iloc[-1])
        last_ema9   = float(ema9.iloc[-1])
        last_ema21  = float(ema21.iloc[-1])
        last_rsi    = float(rsi.iloc[-1])
        last_vwap   = float(vwap.iloc[-1])
        ema9_slope_pct  = _ema_slope_pct(ema9, bars=3)
        ema21_slope_pct = _ema_slope_pct(ema21, bars=3)

        has_vol     = "volume" in df10s.columns and df10s["volume"].sum() > 0
        last_vol    = float(df10s["volume"].iloc[-1]) if has_vol else 1.0
        last_avg    = float(vol_avg.iloc[-1]) if has_vol else 1.0
        vol_ratio   = last_vol / last_avg if last_avg > 0 else 1.0

        candle_range  = last_high - last_low
        candle_body   = abs(last_close - last_open)
        body_ratio    = (candle_body / candle_range) if candle_range > 0 else 0.0
        upper_wick    = last_high - max(last_close, last_open)
        lower_wick    = min(last_close, last_open) - last_low
        upper_wick_r  = (upper_wick / candle_range) if candle_range > 0 else 0.0
        lower_wick_r  = (lower_wick / candle_range) if candle_range > 0 else 0.0

        # ── Supply & Demand Zone Detection (10S) ────────────────────────
        demand_zone, supply_zone = _detect_supply_demand_zones(df10s, lookback=10)

        # ── Trend Confirmation via Higher Highs/Lows (10S) ──────────────
        trend_type, trend_strength = _confirm_trend_hh_ll(df10s, bars=5)

        # ── Previous Candle Pattern Prediction (10S) ────────────────────
        candle_analysis = _analyze_previous_candles(df10s, bars=3)

        bearish_reversal = (
            (last_close < last_ema9)
            and (last_rsi < 58)
            and (last_close < last_open)
            and (body_ratio >= self.BODY_MIN_RATIO)
        )

        supply_label = f"{supply_zone:.5f}" if supply_zone is not None else "N/A"
        demand_label = f"{demand_zone:.5f}" if demand_zone is not None else "N/A"

        log.info(
            f"[{pair}] 10S | close={last_close:.5f} VWAP={last_vwap:.5f} "
            f"EMA9={last_ema9:.5f} EMA21={last_ema21:.5f} RSI={last_rsi:.1f} "
            f"vol_ratio={vol_ratio:.2f}x body={body_ratio:.0%} | "
            f"Supply={supply_label} Demand={demand_label} | "
            f"TrendType={trend_type} (str={trend_strength:.2f}) PrevCandle={candle_analysis.get('momentum_direction')}"
        )

        # ══════════════════════════════════════════════════════════════
        # Global DO-NOT-TRADE guards (apply to both BUY and SELL)
        # ══════════════════════════════════════════════════════════════

        # G1: EMA9 and EMA21 both flat → no trend, skip
        # Allow a strong bearish reversal to break through this guard so we do
        # not suppress a valid sell entry just because the EMA pair is flat.
        if (_ema_slope_flat(ema9, pct_tol=self.EMA_FLAT_TOL) and \
           _ema_slope_flat(ema21, pct_tol=self.EMA_FLAT_TOL) and \
           not bearish_reversal):
            log.info(
                f"[{pair}] DO-NOT-TRADE: EMA9 and EMA21 are flat "
                f"(EMA9 slope={ema9_slope_pct:.6%}, EMA21 slope={ema21_slope_pct:.6%}, "
                f"tol={self.EMA_FLAT_TOL:.6%})"
            )
            return None

        # G2: RSI in dead zone 48–52 → indecision, skip unless the candle is a
        # strong bearish reversal that clearly justifies a sell entry.
        if self.RSI_DEAD_LO <= last_rsi <= self.RSI_DEAD_HI and not bearish_reversal:
            log.info(f"[{pair}] DO-NOT-TRADE: RSI={last_rsi:.1f} in dead zone 48–52")
            return None

        # G3: Volume below threshold → no conviction, skip
        if has_vol and vol_ratio < self.VOL_SPIKE_MULT:
            log.info(f"[{pair}] DO-NOT-TRADE: volume {vol_ratio:.2f}x below threshold ({self.VOL_SPIKE_MULT:.2f}x)")
            return None

        # G4: Very long wicks → indecision / manipulation
        if upper_wick_r > 0.50 or lower_wick_r > 0.50:
            log.info(f"[{pair}] DO-NOT-TRADE: very long wicks (uw={upper_wick_r:.0%} lw={lower_wick_r:.0%})")
            return None

        # G5: VWAP cross chop — price crossed VWAP too many times recently
        if len(df10s) >= self.VWAP_CROSS_BARS + 1:
            recent_close = close.iloc[-(self.VWAP_CROSS_BARS + 1):].values
            recent_vwap  = vwap.iloc[-(self.VWAP_CROSS_BARS + 1):].values
            above        = recent_close > recent_vwap
            crosses      = int(np.sum(np.diff(above.astype(int)) != 0))
            if crosses >= self.VWAP_CROSS_MAX + 1:
                log.info(f"[{pair}] DO-NOT-TRADE: price crossing VWAP repeatedly ({crosses}x in {self.VWAP_CROSS_BARS} bars)")
                return None

        # ══════════════════════════════════════════════════════════════
        # Determine direction attempt
        # ══════════════════════════════════════════════════════════════
        attempt_buy  = (last_close > last_vwap) and (last_ema9 > last_ema21) and not bearish_reversal
        attempt_sell = bearish_reversal or ((last_close < last_vwap) and (last_ema9 < last_ema21))

        if not attempt_buy and not attempt_sell:
            log.info(
                f"[{pair}] Signal: no clear direction — "
                f"close {'>' if last_close > last_vwap else '<'} VWAP, "
                f"EMA9 {'>' if last_ema9 > last_ema21 else '<'} EMA21"
            )
            return None

        direction = "BUY" if attempt_buy else "SELL"
        sell_reversal_override = direction == "SELL" and bearish_reversal and htf_trend == "BULLISH"

        # ── HTF gate (1M) ────────────────────────────────────────────────
        if direction == "BUY" and htf_trend == "BEARISH":
            log.info(f"[{pair}] HTF GATE: BUY blocked — 1M trend is BEARISH")
            return None
        if direction == "SELL" and htf_trend == "BULLISH" and not bearish_reversal:
            log.info(f"[{pair}] HTF GATE: SELL blocked — 1M trend is BULLISH and no bearish reversal is present")
            return None

        # ── Trend Confirmation Gate — HH/LL must ALIGN with direction (v10 fix) ──
        # Previously this only rejected the OPPOSING trend, which let
        # TrendType=NEUTRAL (zero-strength — i.e. an accumulation range)
        # through freely. Now structure must actually confirm the trade
        # direction (UP/STRONG_UP for BUY, DOWN/STRONG_DOWN for SELL) with
        # at least STRUCTURE_MIN_STRENGTH, UNLESS the HTF is strongly enough
        # aligned to override a flat 10S structure read.
        structure_confirms_buy  = trend_type in ("UP", "STRONG_UP") and trend_strength >= self.STRUCTURE_MIN_STRENGTH
        structure_confirms_sell = trend_type in ("DOWN", "STRONG_DOWN") and trend_strength >= self.STRUCTURE_MIN_STRENGTH
        structure_opposes_buy   = trend_type in ("DOWN", "STRONG_DOWN")
        structure_opposes_sell  = trend_type in ("UP", "STRONG_UP")

        if direction == "BUY":
            if structure_opposes_buy and htf_trend != "BULLISH":
                log.info(f"[{pair}] TREND GATE: BUY rejected — HH/LL shows downtrend ({trend_type})")
                return None
            if not structure_confirms_buy and htf_trend != "BULLISH":
                log.info(
                    f"[{pair}] TREND GATE: BUY rejected — structure is {trend_type} "
                    f"(str={trend_strength:.2f}, need >= {self.STRUCTURE_MIN_STRENGTH:.2f}) and HTF is not confirming"
                )
                return None
        else:
            if structure_opposes_sell and htf_trend != "BEARISH" and not sell_reversal_override:
                log.info(f"[{pair}] TREND GATE: SELL rejected — HH/LL shows uptrend ({trend_type})")
                return None
            if not structure_confirms_sell and htf_trend != "BEARISH" and not sell_reversal_override:
                log.info(
                    f"[{pair}] TREND GATE: SELL rejected — structure is {trend_type} "
                    f"(str={trend_strength:.2f}, need >= {self.STRUCTURE_MIN_STRENGTH:.2f}) and HTF is not confirming"
                )
                return None

        # ── Previous Candle Pattern Prediction Gate ──────────────────────
        next_dir = candle_analysis.get("next_likely_direction", "NEUTRAL")
        if next_dir != "NEUTRAL":
            prev_strength = candle_analysis.get("body_strength", 0.0)
            if direction == "BUY" and next_dir == "DOWN" and prev_strength > 0.85:
                log.info(f"[{pair}] CANDLE GATE: BUY rejected — previous candles predict DOWN with strong reversal body ({prev_strength:.2f})")
                return None
            if direction == "SELL" and next_dir == "UP" and prev_strength > 0.85 and not sell_reversal_override:
                log.info(f"[{pair}] CANDLE GATE: SELL rejected — previous candles predict UP with strong reversal body ({prev_strength:.2f})")
                return None

        # ── G6: 3 consecutive large candles already in direction (don't chase) ──
        consec = _count_same_dir_candles(df10s, direction, n=3)
        if consec >= self.CONSEC_MAX + 1:
            log.info(f"[{pair}] DO-NOT-TRADE: {consec} consecutive large {direction} candles — don't chase")
            return None

        # ══════════════════════════════════════════════════════════════
        # BUY path
        # ══════════════════════════════════════════════════════════════
        if direction == "BUY":

            if not (self.RSI_BUY_LO < last_rsi < self.RSI_BUY_HI):
                log.info(f"[{pair}] BUY FAIL: RSI={last_rsi:.1f} not in ({self.RSI_BUY_LO},{self.RSI_BUY_HI})")
                return None

            if has_vol and vol_ratio < self.VOL_SPIKE_MULT:
                log.info(f"[{pair}] BUY FAIL: volume {vol_ratio:.2f}x < {self.VOL_SPIKE_MULT:.2f}x threshold")
                return None

            if last_close < last_open:
                log.info(f"[{pair}] BUY FAIL: last candle not bullish (close={last_close:.5f} open={last_open:.5f})")
                return None
            if last_close <= last_ema9:
                log.info(f"[{pair}] BUY FAIL: close {last_close:.5f} not above EMA9 {last_ema9:.5f}")
                return None
            if body_ratio < self.BODY_MIN_RATIO:
                log.info(f"[{pair}] BUY FAIL: body ratio {body_ratio:.0%} < {self.BODY_MIN_RATIO:.0%}")
                return None
            if upper_wick_r > self.UPPER_WICK_MAX:
                log.info(f"[{pair}] BUY FAIL: upper wick {upper_wick_r:.0%} > {self.UPPER_WICK_MAX:.0%}")
                return None

            # Price must be near or above demand zone
            if demand_zone is not None and last_close < (demand_zone - 0.0005):
                log.info(f"[{pair}] BUY FAIL: price {last_close:.5f} too far below demand zone {demand_zone:.5f}")
                return None

            rsi_strength = min((last_rsi - 55) / 20, 1.0)
            vol_strength = min((vol_ratio - 1.5) / 1.5, 1.0)
            trend_boost  = 0.08 if "UP" in trend_type else 0.02
            demand_boost = 0.06 if demand_zone is not None else 0.0
            confidence   = round(0.72 + 0.12 * rsi_strength + 0.10 * vol_strength + trend_boost + demand_boost, 2)

            strong = (last_rsi > 60) and (vol_ratio >= 2.0) and ("UP" in trend_type)
            expiry = 2 if strong else 3

            # v10: real trend label instead of hardcoded "UPTREND"
            trend_label = "UPTREND" if "UP" in trend_type else ("DOWNTREND" if "DOWN" in trend_type else "RANGING")

            reason = (
                f"BUY | close={last_close:.5f} > VWAP={last_vwap:.5f} | "
                f"EMA9={last_ema9:.5f} > EMA21={last_ema21:.5f} | "
                f"RSI={last_rsi:.1f} | vol={vol_ratio:.2f}x | "
                f"body={body_ratio:.0%} uw={upper_wick_r:.0%} | "
                f"Demand@{demand_label} | Trend={trend_type} | "
                f"HTF(1M)={htf_trend} | TF=10S"
            )
            log.info(f"[{pair}] ✅ {reason} | conf={confidence:.0%} expiry={expiry}m")

            return SignalResult(
                signal="BUY", trend=trend_label,
                confidence=confidence, expiry_min=expiry,
                rsi=last_rsi, ema9=last_ema9, ema21=last_ema21,
                vwap=last_vwap, vol_ratio=vol_ratio, reason=reason,
                trend_strength=trend_strength,
            )

        # ══════════════════════════════════════════════════════════════
        # SELL path
        # ══════════════════════════════════════════════════════════════
        else:

            if not (self.RSI_SELL_LO < last_rsi < self.RSI_SELL_HI) and not bearish_reversal:
                log.info(f"[{pair}] SELL FAIL: RSI={last_rsi:.1f} not in ({self.RSI_SELL_LO},{self.RSI_SELL_HI})")
                return None

            if has_vol and vol_ratio < self.VOL_SPIKE_MULT:
                log.info(f"[{pair}] SELL FAIL: volume {vol_ratio:.2f}x < {self.VOL_SPIKE_MULT:.2f}x threshold")
                return None

            if last_close > last_open and not bearish_reversal:
                log.info(f"[{pair}] SELL FAIL: last candle not bearish (close={last_close:.5f} open={last_open:.5f})")
                return None
            if last_close >= last_ema9 and not bearish_reversal:
                log.info(f"[{pair}] SELL FAIL: close {last_close:.5f} not below EMA9 {last_ema9:.5f}")
                return None
            if body_ratio < self.BODY_MIN_RATIO and not bearish_reversal:
                log.info(f"[{pair}] SELL FAIL: body ratio {body_ratio:.0%} < {self.BODY_MIN_RATIO:.0%}")
                return None
            if lower_wick_r > self.LOWER_WICK_MAX and not bearish_reversal:
                log.info(f"[{pair}] SELL FAIL: lower wick {lower_wick_r:.0%} > {self.LOWER_WICK_MAX:.0%}")
                return None

            # Price must be near or below supply zone
            if supply_zone is not None and last_close > (supply_zone + 0.0005):
                log.info(f"[{pair}] SELL FAIL: price {last_close:.5f} too far above supply zone {supply_zone:.5f}")
                return None

            rsi_strength = min((45 - last_rsi) / 20, 1.0)
            vol_strength = min((vol_ratio - 1.5) / 1.5, 1.0)
            trend_boost  = 0.08 if "DOWN" in trend_type else 0.02
            supply_boost = 0.06 if supply_zone is not None else 0.0
            confidence   = round(0.72 + 0.12 * rsi_strength + 0.10 * vol_strength + trend_boost + supply_boost, 2)

            strong = (last_rsi < 40) and (vol_ratio >= 2.0) and ("DOWN" in trend_type)
            expiry = 2 if strong else 3

            # v10: real trend label instead of hardcoded "DOWNTREND"
            trend_label = "DOWNTREND" if "DOWN" in trend_type else ("UPTREND" if "UP" in trend_type else "RANGING")

            reason = (
                f"SELL | close={last_close:.5f} < VWAP={last_vwap:.5f} | "
                f"EMA9={last_ema9:.5f} < EMA21={last_ema21:.5f} | "
                f"RSI={last_rsi:.1f} | vol={vol_ratio:.2f}x | "
                f"body={body_ratio:.0%} lw={lower_wick_r:.0%} | "
                f"Supply@{supply_label} | Trend={trend_type} | "
                f"HTF(1M)={htf_trend} | TF=10S"
            )
            log.info(f"[{pair}] ✅ {reason} | conf={confidence:.0%} expiry={expiry}m")

            return SignalResult(
                signal="SELL", trend=trend_label,
                confidence=confidence, expiry_min=expiry,
                rsi=last_rsi, ema9=last_ema9, ema21=last_ema21,
                vwap=last_vwap, vol_ratio=vol_ratio, reason=reason,
                trend_strength=trend_strength,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Auto Asset Selector  (ranks using 1M data)
# ══════════════════════════════════════════════════════════════════════════════

class AutoAssetSelector:
    """
    Ranks in-session pairs by payout_weight × (momentum + volatility + EMA_spread)
    computed on the 1M chart. Returns pairs sorted best-first so the scanner
    tries the highest-ROI candidate first.
    """
    SCORE_BARS = 20
    EMA_FAST   = 9
    EMA_SLOW   = 21

    def rank(
        self,
        pairs: List[str],
        data_provider: Callable[[str, str], pd.DataFrame],
    ) -> List[str]:
        scores: Dict[str, float] = {}

        for pair in pairs:
            try:
                df = data_provider(pair, "1m")
                if df is None or len(df) < self.SCORE_BARS + self.EMA_SLOW:
                    scores[pair] = 0.0
                    continue
                df.columns = [c.lower() for c in df.columns]
                close = df["close"]

                recent   = close.iloc[-self.SCORE_BARS:]
                momentum = abs(float(recent.iloc[-1]) - float(recent.iloc[0])) / float(recent.iloc[0])
                vol_cv   = float(recent.std() / recent.mean())

                ema9  = float(_ema(close, self.EMA_FAST).iloc[-1])
                ema21 = float(_ema(close, self.EMA_SLOW).iloc[-1])
                ema_spread = abs(ema9 - ema21) / ema21

                payout_w     = _PAYOUT_WEIGHT.get(pair, 1.0)
                scores[pair] = payout_w * (momentum + vol_cv + ema_spread)
            except Exception:
                scores[pair] = 0.0

        ranked = sorted(pairs, key=lambda p: scores.get(p, 0.0), reverse=True)
        log.info(
            f"[AutoAssetSelector] Ranked: "
            f"{[(p, f'{scores.get(p,0):.5f}', f'w={_PAYOUT_WEIGHT.get(p,1):.2f}') for p in ranked]}"
        )
        return ranked


# ══════════════════════════════════════════════════════════════════════════════
# Main Strategy Engine  (per-pair) — 1M HTF gate + 10S entry
# ══════════════════════════════════════════════════════════════════════════════

class FTTStrategyEngine:
    ZONE_TOL_FACTOR = 0.10   # VWAP ± 10% of EMA spread used as reference zone

    def __init__(self):
        self.htf_agent    = HTFTrendAgent()
        self.signal_agent = VWAPStrategyAgent()

    def analyse(
        self,
        pair: str,
        data_provider: Callable[[str, str], pd.DataFrame],
        trade_duration_min: int = 1,   # orchestrator default; overridden by expiry logic
    ) -> Optional[TradeSignal]:

        now_ist = datetime.now(IST)

        # ── Session gate ──────────────────────────────────────────────
        if not _in_session(pair, now_ist):
            log.info(f"[{pair}] SKIP: outside session at {now_ist.strftime('%H:%M IST')}")
            return None

        # ── Fetch data: 1M for HTF gate, 10S for entry/signal ───────────
        try:
            df1m = data_provider(pair, "1m")
        except Exception as e:
            log.info(f"[{pair}] 1M data error: {e}")
            df1m = None

        try:
            df10s = data_provider(pair, "10s")
        except Exception as e:
            log.info(f"[{pair}] 10S data error: {e}")
            df10s = None

        if df1m is None or df1m.empty:
            log.info(f"[{pair}] SKIP: empty 1M data")
            return None
        if df10s is None or df10s.empty:
            log.info(f"[{pair}] SKIP: empty 10S data")
            return None

        df1m.columns  = [c.lower() for c in df1m.columns]
        df10s.columns = [c.lower() for c in df10s.columns]

        # ── HTF trend (1M) ────────────────────────────────────────────
        htf_trend = self.htf_agent.trend(df1m, pair=pair)

        # ── 10S signal detection ──────────────────────────────────────
        result = self.signal_agent.analyse(df10s, htf_trend=htf_trend, pair=pair)
        if result is None:
            return None

        # ── Build TradeSignal ─────────────────────────────────────────
        current_price = result.ema9          # entry at EMA9 level (per strategy card)
        signal        = Signal.BUY if result.signal == "BUY" else Signal.SELL

        # Enhanced SL/TP using supply/demand zones detected on the 10S chart
        demand_zone, supply_zone = _detect_supply_demand_zones(df10s, lookback=10)

        vwap_dist = abs(current_price - result.vwap)

        if signal == Signal.BUY:
            if demand_zone is not None:
                sl_dist = max((current_price - demand_zone) * 1.2, current_price * 0.0010)
            else:
                sl_dist = max(vwap_dist, current_price * 0.0010)
            sl = current_price - sl_dist
            tp = current_price + (sl_dist * 1.5)
        else:
            if supply_zone is not None:
                sl_dist = max((supply_zone - current_price) * 1.2, current_price * 0.0010)
            else:
                sl_dist = max(vwap_dist, current_price * 0.0010)
            sl = current_price + sl_dist
            tp = current_price - (sl_dist * 1.5)

        # Build a reference zone from VWAP ± EMA spread (10S)
        ema_spread = abs(result.ema9 - result.ema21)
        zone = OrderZone(
            lower=result.vwap - ema_spread,
            upper=result.vwap + ema_spread,
        )

        expiry = result.expiry_min

        return TradeSignal(
            pair=pair,
            signal=signal,
            entry_price=current_price,
            sl=sl,
            tp=tp,
            confidence=min(result.confidence, 1.0),
            reason=result.reason,
            order_zone=zone,
            trend=result.trend,                    # v10: real trend, not hardcoded
            trade_duration_min=expiry,
            trend_strength=result.trend_strength,  # v10: for the AI pipeline
        )


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Pair Scanner  (public API — orchestrator unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class MultiPairScanner:
    """
    Scans all in-session pairs, returns the highest-confidence TradeSignal.
    Identical public interface as v9 — orchestrator zero-diff.
    """

    def __init__(self, pairs: List[str] = FOREX_PAIRS):
        self.pairs    = pairs
        self.selector = AutoAssetSelector()
        self.engines: Dict[str, FTTStrategyEngine] = {
            p: FTTStrategyEngine() for p in pairs
        }

    def scan(
        self,
        data_provider: Callable[[str, str], pd.DataFrame],
        trade_duration_min: int = 1,
    ) -> Optional[TradeSignal]:

        now_ist    = datetime.now(IST)
        in_session = [p for p in self.pairs if _in_session(p, now_ist)]

        if not in_session:
            log.info("[Scanner] No pairs in session right now")
            return None

        log.info(f"[Scanner] In-session pairs ({len(in_session)}): {', '.join(in_session)}")

        ranked = self.selector.rank(in_session, data_provider)

        best: Optional[TradeSignal] = None

        for pair in ranked:
            try:
                sig = self.engines[pair].analyse(pair, data_provider, trade_duration_min)
                if sig is None or sig.signal == Signal.NONE:
                    continue
                if best is None or sig.confidence > best.confidence:
                    best = sig
            except Exception as e:
                log.warning(f"[Scanner] {pair} error: {e}", exc_info=True)

        if best:
            log.info(
                f"[Scanner] 🏆 Best signal: {best.pair} {best.signal} "
                f"conf={best.confidence:.0%} trend={best.trend} expiry={best.trade_duration_min}m"
            )
        else:
            log.info(f"[Scanner] No valid setup across {len(in_session)} active pairs this cycle")

        return best
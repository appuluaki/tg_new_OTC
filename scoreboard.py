"""
scoreboard.py — Session Trade Scoreboard
=========================================
Drop into the project root alongside multi_agent_orchestrator.py.
Import and use inside the orchestrator.

Usage in multi_agent_orchestrator.py:

    from scoreboard import SessionScoreboard

    class MultiAgentTradingSystem:
        def __init__(self, ...):
            ...
            self.board = SessionScoreboard()

        async def _scan_and_trade(self):
            ...
            # After: log.info("[Orchestrator] ✅ Trade placed ...")
            self.board.record_placed(
                pair=best.pair, signal=best.signal.value,
                entry=best.entry_price, sl=best.sl, tp=best.tp,
                duration_s=best.trade_duration_min * 60,
            )

        async def _resolve_trades(self):
            ...
            # Replace: log.info("[Orchestrator] 🔔 Trade expired ...")
            # With:
            result, pnl = await self._get_result(pair, signal, trade_amount)
            self.board.record_result(pair=pair, signal=signal,
                                     result=result, pnl=pnl)

        async def _get_result(self, pair, signal, amount):
            \"\"\"Try to read result from platform, fallback to UNKNOWN.\"\"\"
            if self.exec_agent.bot:
                try:
                    res = await self.exec_agent.bot.get_last_trade_result()
                    if res in ("WIN", "LOSS"):
                        pnl = amount * 0.82 if res == "WIN" else -amount
                        return res, pnl
                except Exception:
                    pass
            return "UNKNOWN", 0.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pytz

log = logging.getLogger("scoreboard")
IST = pytz.timezone("Asia/Kolkata")


@dataclass
class TradeRecord:
    idx:        int
    pair:       str
    signal:     str
    entry:      float
    sl:         float
    tp:         float
    placed_at:  float        # time.monotonic()
    placed_ist: str          # human-readable IST time
    duration_s: int
    result:     str = "PENDING"   # WIN / LOSS / UNKNOWN / PENDING
    pnl:        float = 0.0
    closed_ist: str = ""


class SessionScoreboard:
    """
    Tracks every trade this session.
    Prints a running win/loss table in the terminal after every event.
    """

    RESULT_ICON = {"WIN": "✅", "LOSS": "❌", "UNKNOWN": "❓", "PENDING": "⏳"}

    def __init__(self):
        self.trades:      List[TradeRecord] = []
        self.session_pnl: float             = 0.0
        self._pending:    Dict[str, TradeRecord] = {}

    def _now_ist(self) -> str:
        return datetime.now(IST).strftime("%H:%M:%S")

    def record_placed(
        self,
        pair:       str,
        signal:     str,
        entry:      float,
        sl:         float,
        tp:         float,
        duration_s: int = 60,
    ) -> None:
        idx = len(self.trades) + 1
        rec = TradeRecord(
            idx        = idx,
            pair       = pair,
            signal     = signal,
            entry      = entry,
            sl         = sl,
            tp         = tp,
            placed_at  = time.monotonic(),
            placed_ist = self._now_ist(),
            duration_s = duration_s,
        )
        self.trades.append(rec)
        self._pending[pair] = rec

        log.info(
            f"[Scoreboard] 📋 #{idx} PLACED | {pair} {signal} | "
            f"entry={entry:.5f} | sl={sl:.5f} | tp={tp:.5f} | "
            f"dur={duration_s}s | {rec.placed_ist} IST"
        )

    def record_result(
        self,
        pair:   str,
        signal: str = "",
        result: str = "UNKNOWN",
        pnl:    float = 0.0,
    ) -> None:
        rec = self._pending.pop(pair, None)
        if rec is None:
            for t in reversed(self.trades):
                if t.pair == pair and t.result == "PENDING":
                    rec = t
                    break

        if rec:
            rec.result     = result
            rec.pnl        = pnl
            rec.closed_ist = self._now_ist()
            self.session_pnl += pnl

        icon = self.RESULT_ICON.get(result, "❓")
        log.info(
            f"[Scoreboard] {icon} #{rec.idx if rec else '?'} RESULT | "
            f"{pair} {signal or (rec.signal if rec else '')} | "
            f"{result} | PnL={pnl:+.2f}"
        )
        self._print_table()

    def _print_table(self) -> None:
        """Print full session trade table + summary."""
        total  = len(self.trades)
        wins   = sum(1 for t in self.trades if t.result == "WIN")
        losses = sum(1 for t in self.trades if t.result == "LOSS")
        unk    = sum(1 for t in self.trades if t.result == "UNKNOWN")
        pend   = sum(1 for t in self.trades if t.result == "PENDING")
        wr     = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

        sep = "═" * 74
        log.info(sep)
        log.info(
            f"  SESSION SCOREBOARD  ─  {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}"
        )
        log.info(sep)
        log.info(
            f"  {'#':<4} {'PAIR':<14} {'SIG':<5} {'ENTRY':<10} "
            f"{'RESULT':<9} {'PnL':>7}  {'TIME'}"
        )
        log.info("  " + "─" * 70)

        for t in self.trades:
            icon   = self.RESULT_ICON.get(t.result, "❓")
            pnl_s  = f"${t.pnl:+.2f}" if t.result not in ("PENDING", "UNKNOWN") else "  —  "
            log.info(
                f"  {t.idx:<4} {t.pair:<14} {t.signal:<5} {t.entry:<10.5f} "
                f"{icon} {t.result:<7} {pnl_s:>7}  {t.placed_ist}"
            )

        log.info("  " + "─" * 70)

        # Win/loss bar
        bar = "█" * wins + "░" * losses + "·" * unk + "⋯" * pend
        log.info(
            f"  Total={total}  Win={wins}  Loss={losses}  "
            f"Unknown={unk}  Pending={pend}"
        )
        log.info(f"  Win Rate = {wr:.1f}%  │  Session PnL = ${self.session_pnl:+.2f}")
        log.info(f"  [{bar}]")
        log.info(sep)

    def print_summary(self) -> None:
        """Can be called anytime to print current state."""
        self._print_table()
"""
agents/data_provider_otc.py — Browser tick buffer for OTC pairs
================================================================
Replaces yfinance for EURUSD-OTC (and any OTC pair) which has no
external market data feed.

How it works:
  1. On every call, reads the current live price from the Olymp Trade
     page via get_current_price() (already implemented in browser.py).
  2. Accumulates ticks in an in-process ring buffer (one per pair+tf).
  3. Builds OHLCV candles from the tick stream — 1M candles by default,
     5M by resampling.

Because the orchestrator already calls get_current_price() in its
_scan_and_trade loop (to refresh _LIVE_PRICE_CACHE), we also accept
a pre-fetched price dict so we never call the browser twice per cycle.

Public API:
    make_otc_provider(page, live_price_cache) -> Callable[[str, str], pd.DataFrame]

    page              : the Playwright Page object (bot._page)
    live_price_cache  : the shared _LIVE_PRICE_CACHE dict from the orchestrator
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── Tick buffer ────────────────────────────────────────────────────────────────

class _TickBuffer:
    """
    Stores (timestamp_sec, price) ticks and builds OHLCV candles on demand.
    One instance per (pair, tf).
    """
    def __init__(self, candle_seconds: int = 60, max_candles: int = 120):
        self.candle_sec  = candle_seconds
        self.max_candles = max_candles
        # deque of (epoch_float, price_float)
        self._ticks: Deque[Tuple[float, float]] = deque(maxlen=max_candles * candle_seconds * 2)

    def push(self, price: float, ts: Optional[float] = None):
        if ts is None:
            ts = time.time()
        self._ticks.append((ts, price))

    def to_ohlcv(self) -> pd.DataFrame:
        if len(self._ticks) < 2:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        ticks = list(self._ticks)
        ts_arr = np.array([t[0] for t in ticks])
        px_arr = np.array([t[1] for t in ticks])

        # Bucket into candle_sec-wide bars
        start   = ts_arr[0] - (ts_arr[0] % self.candle_sec)
        end     = ts_arr[-1]
        edges   = np.arange(start, end + self.candle_sec, self.candle_sec)

        rows = []
        for i in range(len(edges) - 1):
            mask = (ts_arr >= edges[i]) & (ts_arr < edges[i + 1])
            if not mask.any():
                continue
            bucket = px_arr[mask]
            rows.append({
                "ts":     edges[i],
                "open":   float(bucket[0]),
                "high":   float(bucket.max()),
                "low":    float(bucket.min()),
                "close":  float(bucket[-1]),
                "volume": float(len(bucket) * 100),   # proxy volume
            })

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df[["open", "high", "low", "close", "volume"]]
        return df.tail(self.max_candles)


# ── Global tick buffers (shared across calls within the same process) ──────────
_BUFFERS: Dict[str, _TickBuffer] = {}

def _get_buffer(pair: str, candle_sec: int) -> _TickBuffer:
    key = f"{pair}_{candle_sec}"
    if key not in _BUFFERS:
        _BUFFERS[key] = _TickBuffer(candle_seconds=candle_sec, max_candles=120)
    return _BUFFERS[key]


# ── Synthetic seed ─────────────────────────────────────────────────────────────

def _seed_buffer(buf: _TickBuffer, seed_price: float, candle_sec: int, n_candles: int = 60):
    """
    Pre-populate the buffer with synthetic ticks centred on seed_price
    so the strategy engine has enough history to compute EMA/RSI immediately
    on first run. Uses a small random walk so indicators are non-degenerate.
    """
    now      = time.time()
    start_ts = now - n_candles * candle_sec
    rng      = np.random.default_rng(42)
    steps    = n_candles * 4       # 4 ticks per candle
    prices   = seed_price + np.cumsum(rng.normal(0, seed_price * 0.00008, steps))
    times    = np.linspace(start_ts, now - candle_sec, steps)
    for t, p in zip(times, prices):
        buf.push(float(p), float(t))


# ── Main factory ───────────────────────────────────────────────────────────────

def make_otc_provider(
    page,                                    # Playwright Page (bot._page)
    live_price_cache: Dict[str, float],      # shared _LIVE_PRICE_CACHE dict
) -> Callable[[str, str], pd.DataFrame]:
    """
    Returns a data_provider(pair, tf) -> pd.DataFrame callable.

    On every call:
      1. Reads the latest price from live_price_cache[pair] (updated
         by the orchestrator's price refresh loop each scan cycle).
         Falls back to page.title() parse if cache is empty.
      2. Pushes the price into the tick buffer for that pair+tf.
      3. Returns OHLCV built from accumulated ticks.
    """

    async def _fallback_price_from_page(pair: str) -> Optional[float]:
        """Last-resort: parse price from page title or DOM."""
        import re
        try:
            title = await page.title()
            m = re.search(r"(\d{1,2}\.\d{3,6})", title)
            if m:
                v = float(m.group(1))
                if 0.5 <= v <= 5.0:   # EUR/USD range sanity
                    return v
        except Exception:
            pass
        try:
            v = await page.evaluate(r"""() => {
                const selectors = ['.current-price','.bid-price','.asset-price',
                                   '.price-value','[class*="price"]'];
                for (const sel of selectors) {
                    const el = document.querySelector(sel);
                    if (el) {
                        const txt = el.innerText.replace(/[^0-9.]/g,'');
                        const n = parseFloat(txt);
                        if (n >= 0.5 && n <= 5.0) return n;
                    }
                }
                return null;
            }""")
            if v:
                return float(v)
        except Exception:
            pass
        return None

    def provider(pair: str, tf: str) -> pd.DataFrame:
        candle_sec = 300 if tf == "5m" else 60

        # Get latest price — prefer the cache (updated every 30s by orchestrator)
        price = live_price_cache.get(pair)

        if price is None or not (0.5 <= price <= 5.0):
            # Cache miss — return empty so engine waits for next cycle
            import logging
            logging.getLogger("data_provider_otc").warning(
                f"[OTCData] {pair} {tf}: no live price in cache yet — waiting"
            )
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        buf = _get_buffer(pair, candle_sec)

        # Seed on first use so EMA/RSI have enough history immediately
        if len(buf._ticks) == 0:
            _seed_buffer(buf, price, candle_sec, n_candles=80)
            import logging
            logging.getLogger("data_provider_otc").info(
                f"[OTCData] {pair} {tf}: seeded tick buffer at {price:.5f} with synthetic history"
            )

        # Push latest price
        buf.push(price)

        df = buf.to_ohlcv()
        if df.empty or len(df) < 5:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # For 5M tf: resample 1M buffer into 5M bars
        if tf == "5m" and candle_sec == 60:
            # Use a 1M buffer and resample
            buf1m = _get_buffer(pair, 60)
            if len(buf1m._ticks) > 0:
                df1m = buf1m.to_ohlcv()
                if not df1m.empty:
                    df = df1m.resample("5min").agg({
                        "open": "first", "high": "max",
                        "low": "min", "close": "last", "volume": "sum"
                    }).dropna()

        return df

    return provider
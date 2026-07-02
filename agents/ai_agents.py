from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class RegimeDetectorAgent:
    """Front-end regime detector for the revised AI pipeline.

    FIX: BREAKOUT_UP / BREAKOUT_DOWN used to fire off a Bollinger squeeze
    plus a single tick beyond the band — which is exactly what a wick out
    of an accumulation range looks like. It now also requires a minimum
    trend_strength (HH/HL or LH/LL confirmation strength, 0.0-1.0, passed
    in from strategy_engine's _confirm_trend_hh_ll) before calling it a
    breakout instead of a fakeout.
    """

    MIN_BREAKOUT_STRENGTH = 0.35

    def detect_regime(
        self,
        closes,
        highs,
        lows,
        rsi,
        macd_val,
        macd_signal,
        bb_upper,
        bb_mid,
        bb_lower,
        atr,
        trend_strength: float = 0.0,
    ) -> str:
        if len(closes) < 20:
            return "UNKNOWN"

        price = float(closes[-1])
        atr_pct = atr / price if price > 0 else 0.0
        high_vol = atr_pct > 0.0008

        slope = 0.0
        if len(closes) >= 10:
            x = list(range(10))
            y = list(closes[-10:])
            denom = sum((xi - sum(x) / 10.0) ** 2 for xi in x)
            slope = 0.0 if denom == 0 else sum(((xi - sum(x) / 10.0) * (yi - sum(y) / 10.0)) for xi, yi in zip(x, y)) / denom

        macd_bull = macd_val > macd_signal
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0.0
        squeeze = bb_width < 0.005

        # FIX: require trend_strength confirmation, not just squeeze + band poke.
        # A squeeze with near-zero trend_strength is an accumulation range —
        # a single tick beyond the band there is a fakeout wick, not a breakout.
        if squeeze and price > bb_upper and trend_strength >= self.MIN_BREAKOUT_STRENGTH:
            return "BREAKOUT_UP"
        if squeeze and price < bb_lower and trend_strength >= self.MIN_BREAKOUT_STRENGTH:
            return "BREAKOUT_DOWN"
        if squeeze and trend_strength < self.MIN_BREAKOUT_STRENGTH:
            return "RANGING"

        trending_up = macd_bull and slope > 0 and rsi > 50
        trending_down = (not macd_bull) and slope < 0 and rsi < 50

        if trending_up and not high_vol:
            return "BULL_TREND"
        if trending_down and not high_vol:
            return "BEAR_TREND"
        if trending_up and high_vol:
            return "HIGH_VOL_UP"
        if trending_down and high_vol:
            return "HIGH_VOL_DOWN"
        return "RANGING"


class UnifiedTransformerAgent:
    """Lightweight transformer-style feature synthesizer for market structure + MTF relationships.

    FIX: market_structure_score and multi_timeframe_score used to be
    derived purely from the strategy engine's own `confidence` value —
    which meant the "AI pipeline confirmation" was largely restating the
    same number back at itself instead of independently checking price
    structure. They now take a real `trend_strength` (from HH/HL / LH/LL
    confirmation) as the primary input, with confidence only as a minor
    secondary nudge.
    """

    def __init__(self, asset: str):
        self.asset = asset

    def build_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        signal = context.get("signal", "HOLD")
        confidence = float(context.get("confidence", 0.0))
        trend_strength = float(context.get("trend_strength", 0.0))

        # Independent structure read: trend_strength does the heavy lifting,
        # confidence only contributes a small secondary adjustment.
        market_structure_score = max(0.0, min(1.0, 0.35 + 0.55 * trend_strength + 0.10 * confidence))
        multi_timeframe_score = max(0.0, min(1.0, 0.30 + 0.60 * trend_strength + 0.10 * confidence))

        return {
            "asset": self.asset,
            "signal": signal,
            "confidence": confidence,
            "trend_strength": trend_strength,
            "market_structure_score": market_structure_score,
            "multi_timeframe_score": multi_timeframe_score,
        }


class SupplyDemandGeometryAgent:
    """Deterministic geometric supply/demand scoring using impulse → base → zone → score."""

    def score(self, impulse: float, base: float, zone: float, volatility: float) -> float:
        impulse_component = max(0.0, min(1.0, float(impulse)))
        base_component = max(0.0, min(1.0, float(base)))
        zone_component = max(0.0, min(1.0, float(zone)))
        volatility_penalty = max(0.0, min(0.35, float(volatility)))
        score = 0.45 * impulse_component + 0.30 * base_component + 0.25 * zone_component
        return max(0.0, min(1.0, score - volatility_penalty))


class MetaClassifierAgent:
    """Calibrated meta-classifier that outputs win and class-probability estimates."""

    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        trend_score = float(features.get("trend_score", 0.0))
        zone_score = float(features.get("zone_score", 0.0))
        regime = features.get("regime", "RANGING")

        win = 0.40 + 0.30 * trend_score + 0.30 * zone_score
        win += 0.05 if regime in {"BULL_TREND", "BREAKOUT_UP"} else -0.05 if regime in {"BEAR_TREND", "BREAKOUT_DOWN"} else 0.0
        # FIX: explicitly penalize RANGING regardless of trend/zone score —
        # a good zone score inside a range is still a range.
        if regime == "RANGING":
            win -= 0.15
        win = max(0.0, min(0.98, win))

        continuation = 0.45 + 0.30 * trend_score
        reversal = 0.30 + 0.20 * (1.0 - trend_score)
        fakeout = max(0.0, min(1.0, 1.0 - continuation - reversal))

        total = continuation + reversal + fakeout
        if total > 0:
            continuation /= total
            reversal /= total
            fakeout /= total

        return {
            "win_prob": win,
            "continuation_prob": max(0.0, min(1.0, continuation)),
            "reversal_prob": max(0.0, min(1.0, reversal)),
            "fakeout_prob": max(0.0, min(1.0, fakeout)),
        }


class PPOExecutionPolicyAgent:
    """Regime-aware execution policy to avoid trading against the dominant trend.

    FIX: RANGING now blocks trades outright (WAIT) rather than only
    down-weighting confidence to 50%. A range is precisely the condition
    the whole pipeline was built to avoid trading into.
    """

    def decide(self, signal: str, confidence: float, regime: str) -> Tuple[str, float]:
        conf = max(0.0, min(1.0, float(confidence)))

        trend_opposed = (signal == "BUY" and regime in {"BEAR_TREND", "BREAKOUT_DOWN", "HIGH_VOL_DOWN"}) or (
            signal == "SELL" and regime in {"BULL_TREND", "BREAKOUT_UP", "HIGH_VOL_UP"}
        )

        if trend_opposed:
            return "WAIT", conf * 0.15

        # FIX: ranging/unknown regime is a hard block, not a soft discount.
        if regime in {"RANGING", "UNKNOWN"}:
            return "WAIT", conf * 0.20

        if signal == "BUY" and conf >= 0.65:
            return "BUY", conf
        if signal == "SELL" and conf >= 0.65:
            return "SELL", conf

        return "WAIT", conf * 0.4


class RevisedPipelineAgent:
    """End-to-end wrapper for the revised AI pipeline used by the orchestrator.

    FIX: analyze_setup() now actually calls self.regime_detector instead of
    trusting a pre-labeled "regime" string handed in from the orchestrator.
    It requires the raw series (closes, highs, lows, rsi, macd, bollinger
    bands, atr) plus trend_strength in market_context. If those aren't
    available, it falls back to the provided regime hint but flags it as
    unverified via `regime_source`.
    """

    def __init__(self, asset: str = "EURUSD-OTC"):
        self.asset = asset
        self.regime_detector = RegimeDetectorAgent()
        self.transformer = UnifiedTransformerAgent(asset)
        self.zone_engine = SupplyDemandGeometryAgent()
        self.meta_classifier = MetaClassifierAgent()
        self.ppo = PPOExecutionPolicyAgent()

    def analyze_setup(self, signal: str, confidence: float, market_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        market_context = market_context or {}
        trend_strength = float(market_context.get("trend_strength", 0.0))

        # ── Regime: compute for real if raw series are provided ──────────
        series = market_context.get("series")
        regime_source = "hint"
        if series is not None:
            try:
                regime = self.regime_detector.detect_regime(
                    closes=series["closes"],
                    highs=series["highs"],
                    lows=series["lows"],
                    rsi=series["rsi"],
                    macd_val=series["macd_val"],
                    macd_signal=series["macd_signal"],
                    bb_upper=series["bb_upper"],
                    bb_mid=series["bb_mid"],
                    bb_lower=series["bb_lower"],
                    atr=series["atr"],
                    trend_strength=trend_strength,
                )
                regime_source = "computed"
            except Exception:
                regime = market_context.get("regime", "RANGING")
        else:
            regime = market_context.get("regime", "RANGING")

        transformer_features = self.transformer.build_features({
            "signal": signal,
            "confidence": confidence,
            "trend_strength": trend_strength,
        })
        zone_score = self.zone_engine.score(
            impulse=float(market_context.get("impulse", 0.7)),
            base=float(market_context.get("base", 0.6)),
            zone=float(market_context.get("zone", 0.65)),
            volatility=float(market_context.get("volatility", 0.15)),
        )
        meta = self.meta_classifier.predict({
            "trend_score": transformer_features["multi_timeframe_score"],
            "zone_score": zone_score,
            "regime": regime,
        })
        action, policy_conf = self.ppo.decide(signal, confidence, regime)
        final_conf = min(0.99, max(0.05, 0.55 * confidence + 0.45 * meta["win_prob"]))
        if action == "WAIT":
            final_conf *= 0.60
        return {
            "regime": regime,
            "regime_source": regime_source,
            "trend_strength": trend_strength,
            "action": action,
            "confidence": final_conf,
            "policy_confidence": policy_conf,
            "win_prob": meta["win_prob"],
            "continuation_prob": meta["continuation_prob"],
            "reversal_prob": meta["reversal_prob"],
            "fakeout_prob": meta["fakeout_prob"],
        }
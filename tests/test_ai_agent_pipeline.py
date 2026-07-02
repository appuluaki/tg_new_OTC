import numpy as np

from agents.ai_agents import (
    MetaClassifierAgent,
    PPOExecutionPolicyAgent,
    RegimeDetectorAgent,
    SupplyDemandGeometryAgent,
    UnifiedTransformerAgent,
)


def test_regime_detector_agent_detects_bull_trend():
    agent = RegimeDetectorAgent()
    closes = np.linspace(1.10, 1.16, 30)
    highs = closes + 0.0005
    lows = closes - 0.0005
    regime = agent.detect_regime(closes, highs, lows, 60.0, 0.0003, 0.0002, 1.161, 1.155, 1.149, 0.0001)
    assert regime in {"BULL_TREND", "HIGH_VOL_UP", "BREAKOUT_UP", "RANGING"}


def test_meta_classifier_returns_normalized_probabilities():
    agent = MetaClassifierAgent()
    out = agent.predict({"trend_score": 0.8, "zone_score": 0.7, "regime": "BULL_TREND"})
    assert set(out.keys()) >= {"win_prob", "continuation_prob", "reversal_prob", "fakeout_prob"}
    assert abs(out["continuation_prob"] + out["reversal_prob"] + out["fakeout_prob"] - 1.0) < 1e-9


def test_supply_demand_geometry_agent_scores_structures():
    agent = SupplyDemandGeometryAgent()
    score = agent.score(impulse=0.9, base=0.75, zone=0.8, volatility=0.2)
    assert 0.0 <= score <= 1.0


def test_ppo_policy_waits_on_low_confidence():
    agent = PPOExecutionPolicyAgent()
    action, conf = agent.decide("BUY", 0.25, "RANGING")
    assert action == "WAIT"
    assert conf <= 0.30


def test_ppo_policy_blocks_buy_in_downtrend():
    agent = PPOExecutionPolicyAgent()
    action, conf = agent.decide("BUY", 0.90, "BEAR_TREND")
    assert action == "WAIT"
    assert conf <= 0.50


def test_ppo_policy_blocks_sell_in_uptrend():
    agent = PPOExecutionPolicyAgent()
    action, conf = agent.decide("SELL", 0.90, "BULL_TREND")
    assert action == "WAIT"
    assert conf <= 0.50


def test_unified_transformer_agent_uses_multi_timeframe_features():
    agent = UnifiedTransformerAgent("EURUSD-OTC")
    features = agent.build_features({"signal": "BUY", "confidence": 0.72})
    assert "market_structure_score" in features
    assert features["market_structure_score"] >= 0.0

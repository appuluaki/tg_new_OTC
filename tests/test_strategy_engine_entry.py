import pandas as pd

from agents.strategy_engine import VWAPStrategyAgent


def _build_bullish_1m_df():
    rows = []
    base = 1.14000
    for i in range(80):
        base += 0.00002
        rows.append(
            {
                "open": base,
                "high": base + 0.00004,
                "low": base - 0.00004,
                "close": base + 0.00001,
                "volume": 2.2 if i < 70 else 3.0,
            }
        )

    rows[-1]["open"] = 1.14135
    rows[-1]["high"] = 1.14160
    rows[-1]["low"] = 1.14115
    rows[-1]["close"] = 1.14152
    rows[-1]["volume"] = 5.4

    return pd.DataFrame(rows)


def _build_bearish_10s_df():
    rows = []
    base = 1.14280
    for i in range(120):
        base -= 0.000008
        rows.append(
            {
                "open": base,
                "high": base + 0.000012,
                "low": base - 0.000012,
                "close": base - 0.000004,
                "volume": 2.2,
            }
        )

    rows[-3:] = [
        {
            "open": 1.141098,
            "high": 1.141100,
            "low": 1.141090,
            "close": 1.141092,
            "volume": 5.838,
        },
        {
            "open": 1.141180,
            "high": 1.141184,
            "low": 1.141178,
            "close": 1.141179,
            "volume": 3.327,
        },
        {
            "open": 1.141178,
            "high": 1.141182,
            "low": 1.141172,
            "close": 1.141170,
            "volume": 2.417,
        },
    ]

    return pd.DataFrame(rows)


def _build_downtrend_buy_blocker_df():
    rows = [
        {"open": 1.142944, "high": 1.142971, "low": 1.142906, "close": 1.142958, "volume": 2.864777},
        {"open": 1.142926, "high": 1.142959, "low": 1.142895, "close": 1.142931, "volume": 2.568536},
        {"open": 1.142942, "high": 1.142971, "low": 1.142921, "close": 1.142943, "volume": 1.811766},
        {"open": 1.142946, "high": 1.142956, "low": 1.142931, "close": 1.142941, "volume": 2.684435},
        {"open": 1.142954, "high": 1.142975, "low": 1.142926, "close": 1.142941, "volume": 1.745791},
        {"open": 1.142922, "high": 1.142951, "low": 1.142898, "close": 1.142927, "volume": 2.578845},
        {"open": 1.142904, "high": 1.142962, "low": 1.142879, "close": 1.142935, "volume": 2.944091},
        {"open": 1.142922, "high": 1.142946, "low": 1.142887, "close": 1.142905, "volume": 2.360201},
        {"open": 1.142948, "high": 1.142971, "low": 1.142915, "close": 1.142936, "volume": 2.221156},
        {"open": 1.142907, "high": 1.142957, "low": 1.142883, "close": 1.142932, "volume": 2.203296},
        {"open": 1.142955, "high": 1.142978, "low": 1.142917, "close": 1.142933, "volume": 1.654238},
        {"open": 1.142908, "high": 1.142959, "low": 1.142876, "close": 1.142928, "volume": 2.888895},
        {"open": 1.142902, "high": 1.142980, "low": 1.142796, "close": 1.142980, "volume": 2.072265},
        {"open": 1.142923, "high": 1.143011, "low": 1.142878, "close": 1.142980, "volume": 2.369021},
        {"open": 1.142907, "high": 1.142982, "low": 1.142855, "close": 1.142940, "volume": 2.897648},
        {"open": 1.142936, "high": 1.142957, "low": 1.142794, "close": 1.142916, "volume": 1.813792},
        {"open": 1.142935, "high": 1.143020, "low": 1.142901, "close": 1.142910, "volume": 1.952560},
        {"open": 1.142910, "high": 1.142976, "low": 1.142879, "close": 1.142975, "volume": 2.616388},
        {"open": 1.142914, "high": 1.142983, "low": 1.142897, "close": 1.142899, "volume": 2.850001},
        {"open": 1.142905, "high": 1.142995, "low": 1.142872, "close": 1.142983, "volume": 1.906795},
        {"open": 1.142896, "high": 1.142989, "low": 1.142842, "close": 1.142830, "volume": 1.600611},
        {"open": 1.142907, "high": 1.143007, "low": 1.142865, "close": 1.142897, "volume": 2.556803},
        {"open": 1.142907, "high": 1.142983, "low": 1.142830, "close": 1.142855, "volume": 1.796915},
        {"open": 1.142916, "high": 1.142980, "low": 1.142872, "close": 1.142897, "volume": 2.562342},
        {"open": 1.142927, "high": 1.142976, "low": 1.142884, "close": 1.142914, "volume": 2.996207},
        {"open": 1.142935, "high": 1.142995, "low": 1.142883, "close": 1.142928, "volume": 2.597605},
        {"open": 1.142878, "high": 1.142950, "low": 1.142834, "close": 1.142903, "volume": 2.105437},
        {"open": 1.142906, "high": 1.142977, "low": 1.142858, "close": 1.142934, "volume": 1.947298},
        {"open": 1.142958, "high": 1.143045, "low": 1.142900, "close": 1.142930, "volume": 2.770114},
        {"open": 1.142940, "high": 1.143015, "low": 1.142897, "close": 1.142969, "volume": 2.937139},
        {"open": 1.142970, "high": 1.143037, "low": 1.142928, "close": 1.142954, "volume": 1.686925},
        {"open": 1.142951, "high": 1.143032, "low": 1.142920, "close": 1.142978, "volume": 2.442717},
        {"open": 1.142975, "high": 1.143034, "low": 1.142930, "close": 1.142986, "volume": 2.178057},
        {"open": 1.142920, "high": 1.142978, "low": 1.142880, "close": 1.142931, "volume": 2.151430},
        {"open": 1.142911, "high": 1.142964, "low": 1.142887, "close": 1.142938, "volume": 1.918580},
        {"open": 1.142910, "high": 1.142947, "low": 1.142874, "close": 1.142899, "volume": 2.002914},
        {"open": 1.142922, "high": 1.142978, "low": 1.142877, "close": 1.142935, "volume": 2.402416},
        {"open": 1.143200, "high": 1.143300, "low": 1.143120, "close": 1.143180, "volume": 2.0},
        {"open": 1.143180, "high": 1.143280, "low": 1.143100, "close": 1.143160, "volume": 2.0},
        {"open": 1.143160, "high": 1.143260, "low": 1.143080, "close": 1.143140, "volume": 2.0},
        {"open": 1.143140, "high": 1.143240, "low": 1.143060, "close": 1.143120, "volume": 2.0},
        {"open": 1.143120, "high": 1.143220, "low": 1.143040, "close": 1.143130, "volume": 2.0},
    ]
    return pd.DataFrame(rows)


def test_rsi_dead_zone_is_narrowed_for_mid_50s():
    agent = VWAPStrategyAgent()
    assert agent.RSI_DEAD_LO == 48
    assert agent.RSI_DEAD_HI == 52


def test_bullish_setup_with_rsi_53_9_is_not_blocked_by_dead_zone():
    agent = VWAPStrategyAgent()
    df = _build_bullish_1m_df()

    result = agent.analyse(df, htf_trend="BULLISH", pair="EURUSD-OTC")

    assert result is not None
    assert result.signal == "BUY"


def test_bearish_10s_setup_is_not_blocked_by_bullish_htf_gate():
    agent = VWAPStrategyAgent()
    df = _build_bearish_10s_df()

    result = agent.analyse(df, htf_trend="BULLISH", pair="EURUSD-OTC")

    assert result is not None
    assert result.signal == "SELL"


def test_buy_is_blocked_when_downtrend_structure_is_strong_on_10s():
    agent = VWAPStrategyAgent()
    df = _build_downtrend_buy_blocker_df()

    result = agent.analyse(df, htf_trend="BULLISH", pair="EURUSD-OTC")

    assert result is None


def test_sell_can_trigger_on_bearish_reversal_even_if_ema9_above_ema21():
    agent = VWAPStrategyAgent()
    rows = []
    base = 1.14260
    for i in range(120):
        base += 0.000002
        rows.append(
            {
                "open": base,
                "high": base + 0.00001,
                "low": base - 0.00001,
                "close": base + 0.000001,
                "volume": 2.5 + (i % 3) * 0.2,
            }
        )

    rows[-1] = {
        "open": 1.14290,
        "high": 1.14292,
        "low": 1.14284,
        "close": 1.14282,
        "volume": 6.0,
    }

    df = pd.DataFrame(rows)
    result = agent.analyse(df, htf_trend="BULLISH", pair="EURUSD-OTC")

    assert result is not None
    assert result.signal == "SELL"

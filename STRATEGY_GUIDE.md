# TRADING STRATEGY GUIDE — $10,000 Account with $2,000 Daily Loss Management

## EXECUTIVE SUMMARY

**Account Setup:**
- Starting Capital: $10,000
- Daily Loss Limit: $2,000 (20% of account)
- Risk Per Trade: $200 (2% of account)
- Max Trades Per Day: 20 trades
- Max Consecutive Losses Before Limit: 10 losses = $2,000

**Key Rule:** Once daily loss hits $2,000, trading stops for the day (circuit breaker).

---

## STRATEGIC FRAMEWORK (Revised AI Pipeline)

```
┌─────────────────────────────────────────────────────────────────┐
│              REVISED 3-MODEL AI PIPELINE + REGIME DETECTOR      │
└─────────────────────────────────────────────────────────────────┘

        R0: MARKET REGIME DETECTOR
        • Trend / Range / High-Vol classification
        • Adjusts feature interpretation by market state
                ↓
        M1: UNIFIED TRANSFORMER
        • Market structure + multi-timeframe attention
        • Learns cross-timeframe relationships end-to-end
                ↓
        M2: SUPPLY/DEMAND GEOMETRY ENGINE
        • Impulse → Base → Zone → Score
        • Deterministic geometric rules, no CNN dependency
                ↓
        M3: META CLASSIFIER (LightGBM/XGBoost)
        • Learns P(win) from upstream features
        • Outputs continuation / reversal / fakeout probabilities
                ↓
        E1: PPO EXECUTION POLICY
        • BUY / SELL / WAIT / SKIP only on pre-screened setups
        • Not responsible for discovering setups from scratch
```

---

## DETAILED AGENT DESCRIPTIONS

### **R0: MARKET REGIME DETECTOR**

**Purpose:**
- Detect whether the market is trending, ranging, or transitioning before any signal is scored.
- Calibrate downstream interpretation so the same structure signal is not treated identically in every condition.

**What happens:**
- Read the current structure and volatility context.
- Assign a regime label that informs the later transformer and meta-classifier.
- Prevent overreacting to similar pattern shapes when the environment changes.

---

### **M1: UNIFIED TRANSFORMER**

**Purpose:**
- Combine market structure and multi-timeframe analysis into a single attention-driven model.
- Allow the network to learn cross-timeframe relationships directly instead of relying on separate overlapping votes.

**Entry Signal Logic:**
```
IF regime = trend:
   prioritize continuation structure and trend-aligned momentum
ELSE IF regime = range:
   emphasize support/resistance interaction and fakeout resistance
ELSE:
   reduce confidence and wait for a cleaner setup
```

**Risk Management Rules:**
```
Position Size: $200 per trade (Fixed)
Take Profit: 1.5% - 2.0% from entry
Stop Loss: 2.0% from entry
Hold Time: 5 minutes maximum
Time Between Trades: 30 seconds minimum
```

**Daily Circuit Breaker:**
```
IF Daily Loss >= $2,000 THEN
   Close all open trades
   Block new entries
   Log event
   Wait for next trading day
```

---

### **M2: SUPPLY/DEMAND GEOMETRY ENGINE**

**Purpose:**
- Replace the prior CNN-driven zone detector with deterministic geometric rules.
- Score candidate zones through the sequence: impulse → base → zone → score.

**Execution Logic:**
1. Detect an impulse move and identify the base structure.
2. Map the relevant supply or demand zone.
3. Score the zone using geometric quality rules and context.
4. Pass the resulting zone features to the meta-classifier.

---

### **M3: META CLASSIFIER (LightGBM/XGBoost)**

**Purpose:**
- Replace the original weighted voting scheme with a learned tabular model.
- Estimate the probability of a win directly from the fused feature set.

**Output Targets:**
- Continuation probability
- Reversal probability
- Fakeout probability
- Calibrated confidence for entry selection

---

### **E1: PPO EXECUTION POLICY**

**Purpose:**
- Act only as an execution layer after the upstream models have found a setup.
- Choose BUY, SELL, WAIT, or SKIP using the pre-screened candidate information.

**Execution Rule:**
- Do not discover new setups from scratch.
- Only determine the best action when the upstream evidence is already strong enough.
- Maintain a narrower role so that policy noise does not dominate the signal quality.

---

## TRADING DAY EXAMPLE

**Scenario: $10,000 account, $2,000 daily limit, $200 per trade**

```
TIME         | ACTION                      | P&L       | BALANCE   | STATUS
─────────────┼─────────────────────────────┼───────────┼───────────┼──────────
9:00 AM      | A1 Selection: EUR/USD       | -         | $10,000   | Ready
9:05 AM      | A2 Plans entry (UP signal)  | -         | $10,000   | 
9:06 AM      | A3 Executes BUY @ 1.0800    | -         | $10,000   | 
9:11 AM      | A4 Closes @ 1.0816 (+1.5%) | +$200     | $10,200   | WIN ✓
────────────┼─────────────────────────────┼───────────┼───────────┼──────────
9:12 AM      | A2 Plans entry (DOWN signal)| -         | $10,200   | 
9:13 AM      | A3 Executes SELL @ 1.0816  | -         | $10,200   | 
9:18 AM      | A4 Closes @ 1.0796 (+1.5%) | +$200     | $10,400   | WIN ✓
────────────┼─────────────────────────────┼───────────┼───────────┼──────────
9:20 AM      | A2 Plans entry (HIGH VOL)  | -         | $10,400   | 
9:21 AM      | A3 Executes SELL @ 1.0810  | -         | $10,400   | 
9:26 AM      | A4 Closes @ 1.0850 (-2.0%) | -$200     | $10,200   | LOSS ✗
────────────┼─────────────────────────────┼───────────┼───────────┼──────────
11:30 AM     | A1 Selection: GBP/USD       | -         | $10,200   | Ready
...          | Continue trading cycle      | -         | -         | -
────────────┼─────────────────────────────┼───────────┼───────────┼──────────
AFTER 10     | Daily Loss = $2,000         | -$2,000   | $8,000    | 
LOSSES       | CIRCUIT BREAKER ACTIVE      | -         | LOCKED    | ⛔ NO MORE TRADES
```

---

## RISK MANAGEMENT RULES

### Position Sizing Formula:
```
Risk Per Trade = Account Size × Risk Percentage
                = $10,000 × 2%
                = $200

Position Size = Risk Amount / Stop Loss %
              = $200 / 2%
              = $10,000 units (contracts)
```

### Daily Limits:
```
Maximum Daily Loss: $2,000 (20% of account)
Minimum Daily Loss Before Stopping: 10 consecutive losses × $200

Daily Loss Tracker:
├─ Losses 1-3:  -$600   (7 losses left)
├─ Losses 4-6:  -$1,200 (4 losses left)
├─ Losses 7-9:  -$1,800 (1 loss left)
└─ Loss 10:     -$2,000 (STOP - No more trading today)
```

### Profit Taking Rules:
```
Per Trade TP: 1.5% - 2.0%
Per Trade SL: 2.0%

Daily Profit Target: Unlimited
Daily Loss Limit: $2,000 hard stop

If Daily Profit > $1,000: Consider reducing trade size
If Daily Loss > $1,500: Prepare for circuit breaker
```

---

## KEY SUCCESS FACTORS

✅ **DO:**
1. Execute trades only during scheduled A1 selection windows
2. Follow A2 strategy rules (confidence + volatility checks)
3. Respect $200 position size strictly
4. Close all trades at 5-minute mark
5. Track daily loss limit religiously
6. Log all trades for A4 learning

❌ **DON'T:**
1. Deviate from $200 position size
2. Hold trades longer than 5 minutes
3. Ignore the $2,000 daily loss limit
4. Trade outside A1 selection windows
5. Over-leverage or size up
6. Revenge trade after losses

---

## ACCOUNT RECOVERY STRATEGY

**If account drops below $8,000:**

1. **Reduce Position Size:** $200 → $150 per trade
2. **Lower Confidence Threshold:** 65% → 60%
3. **Fewer Trades:** Focus on high-quality signals only
4. **Increase Profit Target:** 1.5% → 2.5% (wait for better setups)

**If account recovers to $12,000:**

1. **Return to Base Settings:** $200 per trade
2. **Increase Daily Limit:** $2,000 → $2,500
3. **Consider A/B Testing:** Try new strategies on 10% of account

---

## PERFORMANCE METRICS TO TRACK

**Daily Metrics:**
```
├─ Total Trades: Count
├─ Winning Trades: Count  
├─ Losing Trades: Count
├─ Win Rate: % of winning trades
├─ Largest Win: $ amount
├─ Largest Loss: $ amount
├─ Average Win Size: $
├─ Average Loss Size: $
└─ Profit Factor: Total Profit / Total Loss
```

**Weekly Metrics:**
```
├─ Total P&L: $
├─ Best Day: $ + date
├─ Worst Day: $ + date
├─ Consecutive Winning Days: #
├─ Consecutive Losing Days: #
└─ Consistency Score: Profit days / Total days
```

**Monthly Metrics:**
```
├─ ROI: % return on $10,000
├─ Win Rate: % overall
├─ Sharpe Ratio: Risk-adjusted return
├─ Max Drawdown: Largest loss from peak
└─ Strategy Performance: Which A2 strategies worked best
```

---

## IMPLEMENTATION CHECKLIST

- [ ] Configure account size in settings.py: $10,000
- [ ] Set daily loss limit: $2,000
- [ ] Set risk per trade: $200
- [ ] Configure A1 selection times: 9:00, 11:30, 2:00, 4:00 PM
- [ ] Define A2 confidence thresholds: 60-65%
- [ ] Set A3 hold time: 5 minutes
- [ ] Enable A4 learning feedback
- [ ] Enable daily circuit breaker at $2,000 loss
- [ ] Log all trades to file
- [ ] Test on 1-2 days before live trading
- [ ] Monitor win rate and adjust if < 40%
- [ ] Prepare recovery strategy if balance drops to $8,000

---

## MONTHLY GOALS

**Conservative Target:** 5% return = +$500/month
- ~20 winning trades at $200 profit each
- ~20 losing trades at -$200 each
- Net: +$500 with 50% win rate

**Target Win Rate:** 50-55%
- Below 40%: Review strategy and reduce size
- Above 60%: Monitor for over-optimization

**Risk Tolerance:** Maximum 20% loss per month ($2,000)
- If account hits $8,000: Reduce to $150 per trade

---

**Last Updated:** 2026-05-27
**Strategy Version:** 1.0
**Status:** Active

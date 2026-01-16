# Parameters Tuning Guide

## Core Indicator Parameters

### RSI (Relative Strength Index)

**Parameter: Length**
- Default: 14
- Range: 5-28
- Effect:
  - Lower (5-10): More sensitive, more signals, higher false signals
  - Higher (20-28): Less sensitive, fewer signals, more reliable
- Tuning: Use 14 for 15m chart

**Parameter: Oversold Level**
- Default: 30
- Range: 20-40
- Effect:
  - Lower (20): More aggressive, catches early reversals
  - Higher (40): Conservative, waits for confirmed reversal
- Recommendation: 25-30 for crypto

**Parameter: Overbought Level**
- Default: 70
- Range: 60-80
- Effect: Mirror of oversold
- Recommendation: 70-75 for crypto

### MACD (Moving Average Convergence Divergence)

**Parameter: Fast (12), Slow (26), Signal (9)**
- These are standard values, rarely need changing
- Only adjust if:
  - Too few signals: Reduce fast to 10, slow to 24
  - Too many false signals: Increase fast to 14
- Backtest after any change

### Bollinger Bands

**Parameter: Period**
- Default: 20
- Range: 10-30
- Effect:
  - Shorter (10-15): Tighter bands, more signals, more noise
  - Longer (25-30): Looser bands, fewer signals, cleaner
- Recommendation: 20 is industry standard

**Parameter: Standard Deviations**
- Default: 2.0
- Range: 1.5-2.5
- Effect:
  - Lower (1.5): Tighter bands, more touches
  - Higher (2.5): Looser bands, less touches
- Recommendation: 2.0 for most cryptocurrencies

**Parameter: Squeeze Threshold**
- Default: 0.3 (30% of normal range)
- Range: 0.2-0.4
- Effect:
  - Lower (0.2): Only extreme squeezes trigger signal
  - Higher (0.4): More liberal squeeze detection
- Higher = More trades but lower quality

### EMA (Exponential Moving Average)

**Parameter: Short EMA**
- Default: 9
- Range: 5-13
- Effect: Trend confirmation
- Keep as is

**Parameter: Long EMA**
- Default: 21
- Range: 20-26
- Effect: Trend direction
- Keep as is

**Parameter: Trend EMA**
- Default: 50
- Range: 40-60
- Effect: Higher timeframe equivalent
- Recommendation: 50

## Risk Management Parameters

### Position Size

**Formula**: (Account Size * Risk %) / Stop Loss Distance

**Parameter: Risk Percent**
- Conservative: 1.0-1.5%
- Moderate: 2.0-2.5%
- Aggressive: 3.0-5.0%
- Recommendation: Start with 2%

**Calculate Before Trading**:
```
Example Setup:
Account: 10,000 USDT
Risk: 2% = 200 USDT
Stop Loss: 150 pips = 150 USDT (if 1 point = 1 USDT)

Position Size = 200 / 150 = 1.33 contracts
```

### Stop Loss

**Parameter: Stop Loss Percent**
- Default: 1.5%
- Range: 1.0-2.5%
- Effect:
  - Tighter (1.0%): Less loss per trade, higher exit rate
  - Wider (2.5%): More room to breathe, fewer exits
- Crypto volatility suggests 1.2-1.8%

**Dynamic Stop (ATR-based)**
- Adaptive to market volatility
- Formula: Close - (ATR x Multiplier)
- Multiplier: 1.0-2.0
- Advantage: Adjusts to market conditions

### Take Profit

**Parameter: Take Profit Percent**
- Default: 3.0%
- Range: 2.0-5.0%
- Effect:
  - Conservative (2.0%): More frequent wins
  - Aggressive (5.0%): Larger wins, lower frequency
- Recommended Ratio: TP : SL = 2:1 minimum

### Risk-Reward Ratio

**Definition**: Take Profit / Stop Loss

**Recommended Ratios**:
- Minimum: 1.0 (equal risk and reward)
- Standard: 1.5-2.0 (preferred)
- Aggressive: 2.0-3.0

**Why matters**:
- RR 1.5 with 65% win rate = Profitable
- RR 1.0 with 60% win rate = Barely profitable

## Entry Condition Parameters

### Minimum Conditions to Meet

**Options: 2, 3, or 4**
- Conservative (4): All indicators must agree
- Moderate (3): Most indicators agree
- Aggressive (2): Any 2 indicators trigger

**Recommendations**:
- Trending market: Use 2-3
- Choppy market: Use 3-4
- Highly volatile: Use 4

### Enable Trend Filter

**Conservative Mode (True)**:
- Only trade in confirmed trend
- Avoid counter-trend trades
- Lower success rate but higher win rate

**Aggressive Mode (False)**:
- Trade any reversal
- More signals, more risk
- Higher frequency

### Enable Volume Filter

**Effect**: Only trade when volume > 1.1x 20-day average

**Conservative**:
- Require 1.3x volume
- Better confirmation
- Fewer trades

**Aggressive**:
- Require 1.0x volume
- More trades
- Lower confirmation

## Advanced Combo Features

### Enable Combo Features

Automatically generated from OHLC:
- Body Ratio
- Wick Ratios
- Volume Acceleration
- Pattern Recognition

**Impact**: +3-5% accuracy improvement

### Enable Volume Profile

Analyze volume at different price levels:
- Point of Control (POC)
- Volume Nodes
- Volume Imbalances

**Impact**: +2-3% accuracy on support/resistance

## Trading Rules Parameters

### Max Daily Trades
- Conservative: 10-12
- Moderate: 15-20
- Aggressive: 20-30
- Prevents overtrading

### Max Concurrent Positions
- Conservative: 1
- Moderate: 2-3
- Aggressive: 4-5
- Recommendation: Start with 1

### Min Trade Interval
- Default: 15 minutes
- Prevents "revenge trading"
- Keep between 10-30 minutes

### Daily Loss Limit
- Conservative: 2-3%
- Moderate: 3-5%
- Aggressive: 5-8%
- Stop trading when reached

## Optimization Checklist

### For Trend-Heavy Markets

```json
{
  "rsi_overbought": 65,
  "rsi_oversold": 35,
  "enable_trend_filter": true,
  "stop_loss_percent": 1.2,
  "take_profit_percent": 2.5
}
```

### For Volatile Markets

```json
{
  "rsi_overbought": 75,
  "rsi_oversold": 25,
  "enable_trend_filter": false,
  "stop_loss_percent": 2.0,
  "take_profit_percent": 4.0
}
```

### For Choppy Markets

```json
{
  "bollinger_squeeze_threshold": 0.25,
  "min_conditions_met": 4,
  "enable_volume_filter": true,
  "stop_loss_percent": 1.5,
  "take_profit_percent": 3.5
}
```

## Parameter Testing Workflow

### Step 1: Baseline Test
- Use default parameters
- Backtest 500+ candles
- Record results (Win%, Profit Factor, Drawdown)

### Step 2: Sensitivity Analysis
- Change ONE parameter at a time
- Retest on same data
- Record results

### Step 3: Optimization
- Keep changes that improve results
- Revert that degrade results
- Document improvements

### Step 4: Validation
- Test on out-of-sample data
- Test on different cryptocurrencies
- Test on different timeframes

## When to Change Parameters

### Keep Default If
- Backtest results solid (65%+ win rate)
- Results stable across different periods
- No obvious regime changes

### Adjust If
- Win rate drops below 60%
- Too many consecutive losses
- No trades generated for extended period
- Profit factor below 1.2

### Reload Default If
- New market regime detected
- Strategy stops working
- Major market event occurred

## Common Parameter Mistakes

Don't:
- Change multiple parameters at once
- Over-optimize on single symbol
- Use overly tight stops (< 0.8%)
- Set take profit = stop loss
- Optimize based on one trade

Do:
- Test thoroughly (100+ trades)
- Change one variable at a time
- Use consistent risk:reward ratios
- Validate on fresh data
- Document all changes

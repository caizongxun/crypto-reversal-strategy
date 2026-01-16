# Setup Guide - Crypto Reversal Strategy

## Quick Start (5 minutes)

### For TradingView Pine Script

1. Go to https://www.tradingview.com
2. Open any cryptocurrency chart (BTCUSDT recommended)
3. Set timeframe to 15 minutes
4. Click Pine Script Editor (bottom right)
5. Click New Script
6. Copy entire content from `pine_scripts/crypto_reversal_v1.0.pine`
7. Click Add to Chart
8. Configure parameters in indicator panel

### For Python Backtesting

```bash
cd python_tools
pip install -r requirements.txt
python feature_generator.py
```

## System Requirements

### TradingView
- Any modern web browser
- Internet connection
- TradingView account (free tier acceptable)

### Python Environment
- Python 3.8 or higher
- pip package manager
- 2GB RAM minimum
- 500MB disk space

## Configuration Overview

Three pre-built configurations available:

### Conservative Profile
Best for: Capital preservation, new traders
- Win Rate Target: 72%+
- Monthly Target: 5-8%
- Max Drawdown: 2%

Usage:
```bash
cp configs/conservative_config.json configs/my_config.json
```

### Default Profile
Best for: Balanced approach
- Win Rate Target: 68%
- Monthly Target: 8-12%
- Max Drawdown: 3%

### Aggressive Profile
Best for: Experienced traders
- Win Rate Target: 62-65%
- Monthly Target: 15-20%
- Max Drawdown: 4%

## Parameter Configuration

Key parameters to customize:

```json
{
  "position_size_percent": 2.0,        # Risk per trade
  "stop_loss_percent": 1.5,            # Stop distance
  "take_profit_percent": 3.0,          # Profit target
  "rsi_oversold": 30,                  # Buy threshold
  "rsi_overbought": 70,                # Sell threshold
  "bollinger_band_period": 20,         # BB calculation
  "enable_trend_filter": true,         # Require trend
  "max_daily_trades": 20               # Daily limit
}
```

## Backtesting Workflow

### Step 1: Prepare Historical Data

Download OHLCV data from Binance or other sources:

```bash
# Format: Open,High,Low,Close,Volume
BTCUSDT_15m_2024.csv
```

### Step 2: Run Feature Analysis

```bash
python python_tools/feature_generator.py \
  --file BTCUSDT_15m_2024.csv \
  --lookback 500 \
  --output features.csv
```

### Step 3: Backtest on TradingView

1. Add strategy to chart
2. Click Strategy Tester (bottom right)
3. Configure date range
4. Click Start
5. Review results

### Step 4: Validate Results

Checklist:
- Win rate 60%+
- Profit factor > 1.3
- Max drawdown < 5%
- Sharpe ratio > 0.8

## Live Trading Setup

### Pre-Trading Checklist

- [ ] Backtest minimum 3 months
- [ ] Forward test minimum 2 weeks
- [ ] Start with 50% normal position size
- [ ] Use demo account first
- [ ] Set daily loss limit (2% of account)
- [ ] Have emergency stop procedure ready

### Position Sizing Formula

```
Position Size = (Account Size * Risk %) / Stop Loss Pips

Example:
Account: 10,000 USDT
Risk: 2%
Stop: 150 pips

Position = (10,000 * 0.02) / 150 = 1.33 contracts
```

### Risk Management Rules

1. Never risk more than 2% per trade
2. Max 3 concurrent positions
3. Stop trading after -3% daily loss
4. Never use more than 50% account leverage
5. Keep 20% reserve for emergency

## Troubleshooting

### No Signals Generated

Cause: Market conditions not aligned
Solution:
1. Reduce Bollinger Band squeeze threshold
2. Lower RSI extreme values (75/25 instead of 70/30)
3. Disable trend filter temporarily
4. Check if market is choppy (sideways)

### Too Many Losing Trades

Cause: Parameter mismatch or market regime change
Solution:
1. Tighten stop loss by 0.3%
2. Increase take profit by 0.5%
3. Add multi-timeframe confirmation
4. Retrain features with fresh data

### Strategy Underperforms in Live Trading

Common causes:
1. Slippage and spreads not accounted for
2. Backtest used different market data
3. Trading hours affect liquidity
4. News events create spikes

Solution:
1. Add 0.1-0.2% slippage assumption
2. Reduce position size by 30%
3. Trade during high-volume hours
4. Enable news event filter

## Advanced Configuration

### Using Multiple Timeframes

For better confirmation:
1. Use 15m for entry
2. Use 1h for trend confirmation
3. Use 4h for regime filter

### Feature Tuning

To discover best features:

```bash
python python_tools/feature_generator.py \
  --file data.csv \
  --output features.csv

# Analyze top features in output
```

### Model Retraining

Every month:
1. Collect new 500+ candles
2. Run feature generator
3. Retrain importance rankings
4. Update configuration

## Support & Community

- GitHub Issues: Report bugs
- Discussions: Share configurations
- Backtest Results: Post in Issues

## Disclaimer

This strategy is educational only. Past performance does not guarantee future results. 
Always start with small position sizes and proper risk management. Never risk capital 
you cannot afford to lose.

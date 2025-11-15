# Extended Backtests (AAPL & TSLA)

Parameters: stop-loss 0.5%, take-profit 1.0% per trade.

| Symbol | Model | Trades | Total PnL | Avg PnL | Win Rate | Sharpe |
|--------|-------|--------|-----------|---------|----------|--------|
| AAPL   | RandomForest | 167 | 0.99 | 0.0059 | 0.51 | 11.94 |
| AAPL   | Hybrid        | 167 | 0.99 | 0.0059 | 0.51 | 11.94 |
| TSLA   | RandomForest | 14  | 0.12 | 0.0086 | 0.86 | 8.83 |
| TSLA   | Hybrid        | 14  | 0.12 | 0.0086 | 0.86 | 8.83 |
| BTC/USD| RandomForest | 688 | 8.36 | 0.0122 | 0.81 | 42.73 |
| BTC/USD| Hybrid        | 688 | 8.36 | 0.0122 | 0.81 | 42.73 |
| ETH/USD| RandomForest | 589 | 5.00 | 0.0085 | 0.58 | 25.06 |
| ETH/USD| Hybrid        | 589 | 5.00 | 0.0085 | 0.58 | 25.06 |

> Identical stats indicate the current signal generation (sign of predicted move) remains symmetrical between baseline and hybrid models. Future iterations should incorporate predictive uncertainty, stop-loss triggers based on high/low data, or different holding periods to differentiate the strategies—especially for the crypto pairs where risk overlays cap every trade.

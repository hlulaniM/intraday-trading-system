# Extended Backtests (AAPL & TSLA)

| Symbol | Model | Trades | Total PnL | Avg PnL | Win Rate | Sharpe |
|--------|-------|--------|-----------|---------|----------|--------|
| AAPL   | RandomForest | 167 | 45631.05 | 273.24 | 1.00 | 3596.69 |
| AAPL   | Hybrid        | 167 | 45631.05 | 273.24 | 1.00 | 3596.69 |
| TSLA   | RandomForest | 14  | 5648.54  | 403.47 | 1.00 | 4184.66 |
| TSLA   | Hybrid        | 14  | 5648.54  | 403.47 | 1.00 | 4184.66 |

> The current trade logic (sign of predicted move) produces identical signals for baseline and hybrid models in this small sample; future iterations should incorporate execution rules (stop-loss, trailing exits) to differentiate strategies.

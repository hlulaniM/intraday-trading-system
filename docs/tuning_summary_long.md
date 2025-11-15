# Long-Horizon Tuning Summary

| Symbol | Dropout | Heads | Seq Len | Dir Acc | Level MAE |
|--------|---------|-------|---------|---------|-----------|
| AAPL   | 0.2     | 4     | 60      | 1.00    | 0.53      |
| TSLA   | 0.2     | 4     | 60      | 1.00    | 384.69    |
| BTC/USD| 0.2     | 4     | 60      | 1.00    | 9138.38   |
| ETH/USD| 0.2     | 4     | 60      | 1.00    | 68.67     |

All four symbols currently prefer the shortest sequence (60) with low dropout and four heads; future sweeps can expand around these settings or incorporate additional regularization to differentiate assets.

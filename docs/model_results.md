# Model Results (Phase 3/4)

## Baselines (AAPL latest)
- ARIMA: MAE 0.716, RMSE 0.732
- RandomForest: MAE 0.902, RMSE 0.920
- LSTM: MAE 251.4, RMSE 251.4

## Baselines (TSLA latest)
- ARIMA: MAE 3.30, RMSE 3.33
- RandomForest: MAE 1.75, RMSE 1.81
- LSTM: MAE 369.2, RMSE 369.2

## Hybrid (AAPL latest)
- Direction accuracy: 1.0
- Level MAE: 237.8, RMSE: 237.8, MAPE: 0.873

## Tuning Grid Summary
See `data/processed/sequences/aapl_tuning.json` for dropout/head/sequence variations. Notable best MAE ≈ 257.8 (dropout 0.3, heads 8, seq_len 90).

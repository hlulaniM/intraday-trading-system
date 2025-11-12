<!-- c1a54d32-4340-421b-8290-cec4f71e5f4b 1e117965-91ca-4826-84c8-a166aa38edef -->
# Probabilistic Intraday Trading Forecast System - Implementation Plan

## Project Overview

Build a probabilistic deep learning framework for intraday stock price reversion forecasting that predicts:

1. Direction of price movement (up/down)
2. Probability of price reaching key levels (today's open, yesterday's open/close)

**Timeline:** 6 weeks (October-November 2025)
**Target Performance:**

- Directional Accuracy: ≥90%
- MAE: 15-25% lower than baselines
- Latency: <200 ms
- Profit Factor: >1.5

## Phase 1: Environment Setup & Data Infrastructure (Week 1)

### 1.1 Development Environment Setup

- Set up Python 3.11 environment with virtual environment
- Install core dependencies: Pandas, NumPy, Scikit-learn, TensorFlow/PyTorch
- Install ML libraries: pandas_ta (technical indicators), hmmlearn (HMMs), pgmpy (Bayesian Networks)
- Install visualization: Matplotlib, Seaborn, Plotly/Dash
- Set up version control (Git) and project structure
- Configure IDE (VS Code/Jupyter Notebooks)

### 1.2 API Access & Data Sources

- Register and configure Alpaca API account (free tier)
- Set up Yahoo Finance API access (yfinance library)
- Configure Alpha Vantage API (optional, for historical data)
- Test API connectivity and data retrieval
- Set up data storage structure (local Parquet/CSV, cloud backup)

### 1.3 Data Collection Pipeline

- Implement data fetcher for Alpaca API (minute-level OHLCV + VWAP)
- Create data fetcher for Yahoo Finance (backup source)
- Build data validation and cleaning pipeline
- Implement time alignment and synchronization across data sources
- Set up automated data collection scheduler

## Phase 2: Data Preprocessing & Feature Engineering (Week 1-2)

### 2.1 Data Preprocessing

- Implement outlier detection and filtering (±3σ threshold)
- Handle missing data (forward-fill for market closure, exclude for gaps)
- Implement z-score normalization for numeric features
- Create multi-asset synchronization mechanism
- Build data quality monitoring dashboard

### 2.2 Technical Indicator Engineering

- Implement MACD (Moving Average Convergence Divergence)
- Implement RSI (Relative Strength Index)
- Implement Bollinger Bands
- Implement ATR (Average True Range)
- Implement EMA/SMA (Exponential/Simple Moving Averages)
- Implement Stochastic Oscillator
- Create lagged features to avoid look-ahead bias
- Build feature selection pipeline

### 2.3 Sentiment Analysis Integration

- Set up FinBERT model for financial sentiment classification
- Implement news headline collection (Yahoo Finance/NewsCatcher API)
- Build sentiment scoring pipeline (positive/neutral/negative)
- Implement time-alignment of sentiment with price data
- Create sentiment smoothing (exponential moving average)
- Integrate sentiment scores as model features

### 2.4 Feature Dataset Construction

- Merge OHLCV, technical indicators, and sentiment features
- Create sequence windows (60 time steps for intraday horizon)
- Implement train/validation/test split (70/20/10, chronological)
- Save processed datasets in Parquet format
- Create feature importance analysis

## Phase 3: Model Development (Week 2-3)

### 3.1 Baseline Models Implementation

- Implement ARIMA baseline model
- Implement Random Forest baseline
- Implement simple LSTM baseline (deterministic)
- Evaluate baseline performance metrics (MAE, RMSE, accuracy)

### 3.2 Core Model Architecture

- **LSTM Layers:** Implement for short-term temporal dependencies
- Design LSTM network with appropriate hidden units
- Add dropout regularization (0.3-0.5)
- Implement sequence-to-sequence architecture

- **Transformer Encoder:** Implement for long-range attention
- Multi-head self-attention mechanism
- Positional encoding for time series
- Feed-forward layers

- **Feature Fusion Layer:** Combine LSTM and Transformer outputs
- Concatenation or weighted combination
- Dense layers for feature integration

### 3.3 Probabilistic Extension

- Implement Monte Carlo Dropout for uncertainty quantification
- Add Bayesian inference head
- Create probability distribution outputs (direction + level-reach)
- Implement confidence interval estimation
- Build calibration metrics (Brier Score)

### 3.4 Model Training

- Implement Adam optimizer with adaptive learning rate (0.001)
- Create combined loss function (MSE + Negative Log-Likelihood)
- Implement early stopping (15 epochs patience)
- Add learning rate scheduling
- Implement batch training (sequence windows of 60 steps)
- Set up GPU training (if available) or cloud training (Google Colab)

## Phase 4: Model Evaluation & Optimization (Week 3-4)

### 4.1 Offline Backtesting

- Implement walk-forward backtesting (sequential, no look-ahead bias)
- Evaluate on unseen historical data (minimum 3 months)
- Calculate performance metrics:
- Directional Accuracy (DA)
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Brier Score (calibration)
- Latency measurements

### 4.2 Model Comparison

- Compare against baselines (ARIMA, Random Forest, deterministic LSTM)
- Statistical significance testing (paired t-test, 95% confidence)
- Performance visualization (calibration curves, error distributions)
- Identify best-performing model configuration

### 4.3 Hyperparameter Optimization

- Implement grid search or Bayesian optimization
- Tune: learning rate, dropout rates, hidden units, sequence length
- Optimize for accuracy-latency trade-off
- Document optimal hyperparameters

### 4.4 Model Compression (if needed for latency)

- Implement model pruning
- Test weight quantization
- Evaluate lightweight alternatives (Temporal Convolutional Networks)
- Measure latency improvements vs. accuracy trade-offs

## Phase 5: Deployment Infrastructure (Week 4-5)

### 5.1 API Development

- Build Flask/FastAPI microservice
- Create RESTful endpoints:
- `/predict` - Get forecast for current market state
- `/health` - System health check
- `/metrics` - Performance metrics
- Implement request validation and error handling
- Add API authentication (if needed)
- Create API documentation

### 5.2 TradingView Integration

- Develop Pine Script plugin for TradingView
- Implement webhook alerts system
- Create custom indicator visualization
- Build real-time data streaming pipeline
- Test integration with TradingView charts

### 5.3 Real-time Data Pipeline

- Implement live data streaming from Alpaca API
- Create real-time feature engineering pipeline
- Build caching mechanism for low latency
- Implement data synchronization checks
- Add monitoring and logging

### 5.4 Dashboard Development

- Build Plotly Dash dashboard for:
- Real-time predictions visualization
- Performance metrics tracking
- Model confidence intervals
- Trading signals display
- Create historical performance charts
- Implement alert system for high-confidence predictions

## Phase 6: Live Testing & Validation (Week 5-6)

### 6.1 Pre-deployment Testing

- Conduct paper trading simulation
- Test system stability and reliability
- Verify latency requirements (<200 ms)
- Test error handling and recovery
- Validate data pipeline robustness

### 6.2 30-Day Live Validation

- Deploy system for continuous 30-day operation
- Monitor daily performance metrics:
- Profit Factor (PF)
- Win Rate (WR)
- Calibration Error (CE)
- Average Response Time
- Log all predictions and outcomes
- Track system availability (target: ≥99%)

### 6.3 Performance Monitoring

- Implement automated logging system
- Create daily performance reports
- Monitor for model drift (if live accuracy deviates >5% from backtest)
- Implement automatic retraining trigger (if needed)
- Track latency and system health

### 6.4 Adaptive Retraining

- Implement online learning mechanism (if needed)
- Create retraining pipeline with latest data
- Test model adaptation to market regime changes
- Validate retrained model performance

## Phase 7: Documentation & Reporting (Week 6)

### 7.1 Technical Documentation

- Document system architecture
- Create API documentation
- Write user guide for TradingView integration
- Document model training procedures
- Create deployment guide

### 7.2 Performance Report

- Compile 30-day live testing results
- Compare live vs. backtest performance
- Analyze prediction accuracy and profitability
- Document limitations and challenges
- Create visualizations and charts

### 7.3 Research Documentation

- Write methodology section
- Document experimental results
- Create comparison tables (baselines vs. proposed model)
- Prepare presentation materials
- Finalize research report/thesis

## Key Deliverables

1. **Working System:**

- Trained probabilistic deep learning model
- Real-time prediction API
- TradingView integration
- Performance dashboard

2. **Performance Metrics:**

- Directional accuracy ≥90%
- MAE 15-25% lower than baselines
- Latency <200 ms
- Profit Factor >1.5
- 30-day live validation results

3. **Documentation:**

- Technical documentation
- API documentation
- Research report
- Code repository (GitHub)

## Risk Mitigation

- **Data Quality Issues:** Implement robust validation and cleaning
- **API Failures:** Have backup data sources (Yahoo Finance)
- **Model Overfitting:** Use cross-validation and regularization
- **Latency Issues:** Implement model compression and edge deployment
- **Market Volatility:** Test across different market conditions
- **System Downtime:** Implement monitoring and automatic recovery

## Success Criteria

- Model achieves target performance metrics
- System runs reliably for 30 days (≥99% uptime)
- Live performance within 5% of backtest results
- Successful TradingView integration
- Complete documentation and reproducible code

### To-dos

- [ ] Check what text extraction tools are available on the system (pdftotext, pandoc, python, etc.)
- [ ] Extract text from CONCEPT_PAPER (4).pdf
- [ ] Extract text from Nkuna HT_214755718_IEEE (1).docx
- [ ] Extract text from prposal (2).pdf
- [ ] Extract text from systematic_literature_review.pdf
- [ ] Read and analyze all extracted text to understand document contents
- [ ] Provide summaries of what each document contains
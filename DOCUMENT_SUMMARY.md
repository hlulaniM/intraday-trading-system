# Document Analysis Summary

## Overview
This folder contains 4 academic documents related to financial market forecasting using machine learning and deep learning techniques, specifically focused on intraday trading and probabilistic price prediction.

---

## 1. CONCEPT_PAPER (4).pdf
**Title:** Market Trade Forecast: A Probabilistic Approach to Price Reversion Analysis

**Author:** Mabunda Hlulani (Student Number: 213067605)
**Institution:** Tshwane University of Technology, Department of Computer Systems Engineering
**Supervisors:** Dr. Oluwasogo Moses Olaifa, Dr. Chunling Du

### Key Content:
- **Research Focus:** Probabilistic approach to predicting stock price reversion to key reference points (today's open, yesterday's open, yesterday's close)
- **Objectives:**
  - Analyze frequency of price returns to important reference points
  - Determine optimal trading times (8 AM, noon, 4 PM)
  - Forecast price movements using past intraday data with emphasis on reversion probabilities
  - Ensure model applicability across different asset classes
  - Validate model in real trading environment for 30 days

- **Methodology:**
  - Machine learning with probabilistic techniques
  - Data sources: Yahoo Finance, Alpha Vantage, Bloomberg
  - Tools: Python, Pandas, NumPy, Scikit-learn, TensorFlow/PyTorch
  - Probabilistic modeling: HMMs, Bayesian Networks, Monte Carlo simulations
  - Visualization: Matplotlib, Seaborn, Plotly/Dash
  - Deployment: Flask/Django API

- **Expected Contributions:**
  - Better forecasting with probability estimates
  - Smarter trading decisions with risk management
  - Real-time flexibility
  - Scalability across markets



### Key Content:
- **Research Focus:** Integration of machine learning and technical analysis in algorithmic trading, specifically for cryptocurrency markets
- **Methodology:**
  - Data source: Alpaca API
  - Model: Long Short-Term Memory (LSTM) networks
  - Technical indicators: MACD, RSI (using pandas_ta library)
  - Sentiment analysis: News headlines with anomaly detection (Isolation Forest)
  - Backtesting: Moving averages strategy

- **Results:**
  - Final portfolio value: $10,011.05 (0.11% return)
  - LSTM model predictions tracked market trends closely
  - System processed Bitcoin and Ethereum minute-level data
  - Sentiment analysis integrated with price data

- **Contribution:** Demonstrates practical application of ML in cryptocurrency trading with real API integration

---

## 3. prposal (2).pdf
**Title:** Market Trade Forecast: A Probabilistic Approach to Price Trending Analysis

**Author:** Mabunda Hlulani (Student Number: 213067605)
**Institution:** Tshwane University of Technology
**Degree:** PGDip. Computer Systems Engineering
**Supervisors:** Dr. Oluwasogo Moses Olaifa, Prof. Chunling Du
**Submission Date:** 31 October 2025

### Key Content:
- **Research Proposal** - Comprehensive research proposal document
- **Abstract:** Proposes intelligent probabilistic deep learning framework for forecasting directional movement and level attainment of asset prices using high-frequency data
- **Key Features:**
  - Bayesian extensions of LSTM and Transformer models
  - Integration of alternative data (sentiment, ESG indicators, volatility dynamics)
  - Asset-agnostic design
  - TradingView integration via Pine Script pipelines
  - Systematic literature review of 260+ primary studies

- **Research Questions:**
  - RQ1: What types of probabilistic and DL models are most suitable?
  - RQ2: How can the model improve forecasting effectiveness?
  - RQ3: What strategies ensure real-time deployment and explainability?
  - RQ4: How to evaluate and validate through 30-day live trading?

- **Methodology:**
  - Design Science Research Methodology (DSRM)
  - Data: Alpaca API (OHLCV + VWAP), sentiment from FinBERT
  - Models: LSTM + Transformer with Bayesian inference (Monte Carlo Dropout)
  - Technical indicators: MACD, RSI, Bollinger Bands, ATR
  - Deployment: Flask/FastAPI microservice with TradingView integration

- **Expected Performance:**
  - Directional Accuracy: ≥90%
  - MAE: 15-25% lower than baselines
  - Latency: <200 ms
  - Profit Factor: >1.5
  - System Availability: ≥99%

- **Timeline:** 6 weeks (October-November 2025)
- **Budget:** R300 (primarily printing costs, using open-source tools)

---

## 4. systematic_literature_review.pdf
**Title:** Market Trade Forecast: A Probabilistic Approach to Price Trending Analysis - Systematic Literature Review

**Authors:** Hlulani Mabunda, Oluwasogo Moses Olaifa, Chunling Du
**Institution:** Tshwane University of Technology

### Key Content:
- **Comprehensive Systematic Literature Review (SLR)**
- **Scope:** 60 peer-reviewed studies (2018-2025) on ML/DL methods for probabilistic forecasting of intraday stock price reversion
- **Methodology:** PRISMA-based approach

- **Research Questions:**
  - RQ1: What types of probabilistic and DL models are used, and how are architectures classified?
  - RQ2: What is the effectiveness difference between techniques?
  - RQ3: What are challenges in computational efficiency, explainability, and real-time deployment?
  - RQ4: How can models be integrated into TradingView, and what are practical implications?

- **Key Findings:**
  - **Model Distribution:** LSTM (30%), Transformer (22%), CNN (17%), Probabilistic (15%), RL (9%)
  - **Performance:** DL models achieve 12-18% accuracy improvement over traditional methods
  - **Accuracy Levels:** LSTM (88-91%), Transformer (93-95%), Hybrid Probabilistic (94-96%)
  - **Challenges:** 
    - Computational latency (42% of studies)
    - Overfitting (23%)
    - Explainability (19%)
    - Real-time deployment (8-10%)

- **Database Search Results:**
  - Initial records: 8,235
  - After deduplication: 7,651
  - After screening: 750
  - Final included: 60 studies

- **Top Journals:**
  - Expert Systems with Applications (38 publications)
  - IEEE Access (26)
  - Applied Soft Computing Journal (17)

- **Geographic Distribution:**
  - China (26%), India (21%), United States (14%), United Kingdom (9%)

- **Key Insights:**
  - Field is methodologically mature but operationally underdeveloped
  - Need for explainable AI, latency optimization, and deployment frameworks
  - Gap between research prototypes and real-world trading systems

---

## Common Themes Across Documents

1. **Probabilistic Forecasting:** All documents emphasize uncertainty quantification and probability-based predictions
2. **Intraday Trading Focus:** Emphasis on short-term, high-frequency trading strategies
3. **Deep Learning:** Heavy use of LSTM, Transformer, and hybrid architectures
4. **Real-time Deployment:** Integration with trading platforms (TradingView, Alpaca API)
5. **Technical Analysis:** Integration of indicators (MACD, RSI, moving averages)
6. **Alternative Data:** Sentiment analysis, news data, ESG indicators
7. **Validation:** 30-day live trading validation periods
8. **Institution:** All from Tshwane University of Technology, Computer Systems Engineering

---

## Document Relationships

- **CONCEPT_PAPER** and **prposal** are by the same author (Mabunda Hlulani) and cover similar research but at different stages (concept vs. full proposal)
- **systematic_literature_review** provides the academic foundation and literature synthesis for the research
- **Nkuna's document** is a separate but related study on algorithmic trading with ML, providing a practical implementation example

---

## Technical Stack (Common Across Documents)

- **Languages:** Python 3.11
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/PyTorch
- **ML Models:** LSTM, GRU, Transformer, CNN-LSTM hybrids
- **Probabilistic:** HMMs, Bayesian Networks, Monte Carlo Dropout
- **Data Sources:** Alpaca API, Yahoo Finance, Alpha Vantage
- **Deployment:** Flask/FastAPI, TradingView Pine Script
- **Visualization:** Matplotlib, Seaborn, Plotly/Dash


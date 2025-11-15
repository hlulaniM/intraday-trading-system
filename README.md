# Probabilistic Intraday Trading Forecast System

A probabilistic deep learning framework for intraday stock price reversion forecasting that predicts both the direction of price movement and the probability of reaching key price levels.

## Project Overview

This system combines:
- **LSTM networks** for short-term temporal dependencies
- **Transformer encoders** for long-range attention
- **Bayesian inference** for uncertainty quantification
- **Real-time integration** with TradingView platform

### Key Features

- Predicts price direction (up/down) with probability estimates
- Estimates probability of price reaching key levels (today's open, yesterday's open/close)
- Real-time inference with <200ms latency
- TradingView integration via Pine Script
- Interactive dashboard for monitoring and visualization

## Target Performance

- **Directional Accuracy:** ≥90%
- **MAE:** 15-25% lower than baselines
- **Latency:** <200 ms
- **Profit Factor:** >1.5

## Project Structure

```
project/
├── src/
│   ├── data/           # Data collection and preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # Model definitions and training
│   ├── api/            # API server code
│   ├── dashboard/      # Dashboard application
│   └── utils/          # Utility functions
├── tests/              # Unit and integration tests
├── notebooks/          # Jupyter notebooks for exploration
├── docs/               # Documentation
├── config/             # Configuration files
├── scripts/            # Deployment and utility scripts
├── requirements.txt    # Python dependencies
├── requirements-dev.txt # Development dependencies
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- Git
- Alpaca API account (free tier available)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PROJECT
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   export ALPACA_API_KEY="your_key"
   export ALPACA_SECRET_KEY="your_secret"
   export ALPACA_BASE_URL="https://paper-api.alpaca.markets/v2"
   ```

5. **Run tests:**
   ```bash
   pytest tests/
   ```

## Development Setup

For development, install additional dev dependencies:

```bash
pip install -r requirements-dev.txt
```

### Code Quality

- **Format code:** `black src/ tests/`
- **Lint code:** `flake8 src/ tests/`
- **Type check:** `mypy src/`
- **Run tests:** `pytest tests/ --cov=src`

## Usage

### Collecting Intraday Data

Use the Phase 2 collection script to download and store raw Alpaca minute bars (defaults are read from `config/config.yaml`):

```bash
source .venv/bin/activate
python scripts/collect_intraday_data.py \
  --symbols AAPL TSLA BTC/USD \
  --lookback-days 5 \
  --timeframe 1Min \
  --format parquet
```

Clean data and build feature sets by importing the new utilities:

```python
from pathlib import Path
import pandas as pd

from data.preprocess import clean_price_frame, save_processed_frame
from features import (
    add_indicators,
    SplitConfig,
    generate_sequences,
    split_sequences,
    save_sequence_pack,
)

df = pd.read_parquet("data/raw/AAPL/...")
clean_df = clean_price_frame(df)
save_processed_frame(clean_df, "AAPL")

feature_df = add_indicators(clean_df, config={"features": {"technical_indicators": {}}})
feature_df = feature_df.dropna()
feature_columns = feature_df.columns.drop(["symbol"])
X, y, feature_names = generate_sequences(
    feature_df,
    feature_columns=feature_columns,
    target_column="close",
    sequence_length=60,
)
splits = split_sequences(X, y, SplitConfig())
save_sequence_pack(splits, basename="aapl_sequences", feature_names=feature_names)
```

### Training Baseline Models

```bash
PYTHONPATH=src python scripts/train_baselines.py \
  data/processed/sequences/aapl_sequences.npz \
  --close-index -1 \
  --output data/processed/sequences/aapl_baselines.json
```

### Training the Hybrid Model

```bash
PYTHONPATH=src python scripts/train_hybrid.py \
  data/processed/sequences/aapl_sequences.npz \
  --config config/config.yaml \
  --close-index -1 \
  --epochs 50 \
  --batch-size 32 \
  --output data/processed/sequences/aapl_hybrid_metrics.json
```

### Hyperparameter Sweep

```bash
PYTHONPATH=src python scripts/tune_hybrid.py \
  data/processed/sequences/aapl_sequences.npz \
  --config config/config.yaml \
  --close-index -1 \
  --epochs 20 \
  --output data/processed/sequences/aapl_hybrid_tuning.json
```

### Training the Model

```bash
python -m src.models.train
```

### Running the API Server

```bash
uvicorn src.api.server:app --reload
```

### Running the Dashboard

```bash
python -m src.dashboard.app
```

## Documentation

- [System Architecture](SYSTEM_ARCHITECTURE.md) - Detailed architecture diagrams
- [API Documentation](docs/api.md) - API endpoint documentation
- [User Guide](docs/user_guide.md) - Usage instructions

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests and linting: `pytest && black . && flake8 .`
4. Commit with conventional commits: `git commit -m "feat: add new feature"`
5. Push and create a pull request

## License

[Add your license here]

## Authors

- Mabunda Hlulani (213067605@tut4life.ac.za)
- Supervisors: Dr. Oluwasogo Moses Olaifa, Prof. Chunling Du

## Acknowledgments

Tshwane University of Technology, Department of Computer Systems Engineering


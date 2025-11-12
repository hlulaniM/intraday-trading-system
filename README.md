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
   cp .env.example .env
   # Edit .env with your API keys
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

### Training the Model

```bash
python -m src.models.train
```

### Running the API Server

```bash
python -m src.api.main
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


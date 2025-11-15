"""Backtest baseline and hybrid models on saved sequence datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import math

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.baselines import train_random_forest_baseline  # noqa: E402
from models.hybrid_model import HybridConfig, HybridForecastModel  # noqa: E402
from models.trainer import SequenceDataset, HybridTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest models on sequence dataset.")
    parser.add_argument("dataset", help="Path to .npz dataset.")
    parser.add_argument("--close-index", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="docs/backtests/backtest_results.json")
    return parser.parse_args()


def simulate_trades(entry: np.ndarray, exit_: np.ndarray, signal: np.ndarray) -> dict:
    pnl = signal * (exit_ - entry)
    cumulative = pnl.cumsum()
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    profit_factor = wins / losses if losses > 0 else float("inf")
    if math.isinf(profit_factor):
        profit_factor = None
    win_rate = float((pnl > 0).mean())
    max_drawdown = float((np.maximum.accumulate(cumulative) - cumulative).max())
    return {
        "trades": len(pnl),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "win_rate": win_rate,
        "profit_factor": float(profit_factor) if profit_factor is not None else None,
        "max_drawdown": max_drawdown,
    }


def backtest_model(entry: np.ndarray, exit_: np.ndarray, direction_signal: np.ndarray) -> dict:
    signal = np.where(direction_signal > 0, 1, -1)
    return simulate_trades(entry, exit_, signal)


def main() -> None:
    args = parse_args()
    dataset = SequenceDataset.from_npz(args.dataset, close_index=args.close_index)
    entry_prices = dataset.last_closes("test")
    true_prices = dataset.test_y

    rf_result = train_random_forest_baseline(
        dataset.train_X,
        dataset.train_y,
        dataset.test_X,
        dataset.test_y,
    )
    rf_signal = np.sign(rf_result.predictions - entry_prices)
    rf_metrics = backtest_model(entry_prices, true_prices, rf_signal)

    trainer = HybridTrainer(dataset, config=HybridConfig())
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    dir_pred, level_pred = trainer.model.model.predict(dataset.test_X, verbose=0)
    hybrid_signal = np.sign(level_pred.flatten() - entry_prices)
    hybrid_metrics = backtest_model(entry_prices, true_prices, hybrid_signal)

    results = {
        "baseline_random_forest": rf_metrics,
        "hybrid": hybrid_metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

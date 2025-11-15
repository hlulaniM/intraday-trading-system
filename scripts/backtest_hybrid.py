"""Backtest baseline and hybrid models across multiple symbols."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.baselines import train_random_forest_baseline  # noqa: E402
from models.hybrid_model import HybridConfig  # noqa: E402
from models.trainer import SequenceDataset, HybridTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest models on sequence datasets.")
    parser.add_argument("datasets", nargs="+", help="Paths to .npz datasets.")
    parser.add_argument("--close-index", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="docs/backtests/")
    parser.add_argument("--summary", default="docs/backtests/summary.json")
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
    sharpe = None
    if len(pnl) > 1:
        std = pnl.std(ddof=1)
        if std > 0:
            sharpe = float((pnl.mean() / std) * np.sqrt(len(pnl)))
    return {
        "trades": len(pnl),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "win_rate": win_rate,
        "profit_factor": float(profit_factor) if profit_factor is not None else None,
        "max_drawdown": max_drawdown,
        "equity_curve": cumulative.tolist(),
        "sharpe": sharpe,
    }


def backtest_model(entry: np.ndarray, exit_: np.ndarray, direction_signal: np.ndarray) -> dict:
    signal = np.where(direction_signal > 0, 1, -1)
    return simulate_trades(entry, exit_, signal)


def run_backtest(dataset_path: Path, close_index: int, epochs: int, batch_size: int) -> dict:
    dataset = SequenceDataset.from_npz(dataset_path, close_index=close_index)
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
    trainer.train(epochs=epochs, batch_size=batch_size)
    _, level_pred = trainer.model.model.predict(dataset.test_X, verbose=0)
    hybrid_signal = np.sign(level_pred.flatten() - entry_prices)
    hybrid_metrics = backtest_model(entry_prices, true_prices, hybrid_signal)

    return {
        "baseline_random_forest": rf_metrics,
        "hybrid": hybrid_metrics,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for dataset in args.datasets:
        path = Path(dataset)
        if not path.exists():
            print(f"Skipping missing dataset: {path}")
            continue
        symbol = path.stem.split("_")[0].upper()
        print(f"Backtesting {symbol} ...")
        results = run_backtest(path, args.close_index, args.epochs, args.batch_size)
        summary[symbol] = results
        output_path = output_dir / f"{symbol.lower()}_backtest.json"
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

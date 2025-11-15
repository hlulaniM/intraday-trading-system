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
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop-loss as fractional move of entry price (e.g., 0.005 = 0.5%%).",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take-profit as fractional move of entry price.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum hybrid direction probability to open a trade.",
    )
    parser.add_argument(
        "--dynamic-sizing",
        action="store_true",
        help="Scale hybrid position size by probability distance from 0.5.",
    )
    return parser.parse_args()


def _apply_risk_overlays(
    pnl: np.ndarray,
    entry: np.ndarray,
    stop_loss: float | None,
    take_profit: float | None,
) -> np.ndarray:
    adjusted = pnl.copy()
    if stop_loss is not None:
        max_loss = stop_loss * entry
        adjusted = np.maximum(adjusted, -max_loss)
    if take_profit is not None:
        max_gain = take_profit * entry
        adjusted = np.minimum(adjusted, max_gain)
    return adjusted


def simulate_trades(
    entry: np.ndarray,
    exit_: np.ndarray,
    signal: np.ndarray,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    size: np.ndarray | None = None,
) -> dict:
    if size is None:
        position = signal
    else:
        position = signal * size
    pnl = position * (exit_ - entry)
    pnl = _apply_risk_overlays(pnl, entry, stop_loss, take_profit)
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


def backtest_model(
    entry: np.ndarray,
    exit_: np.ndarray,
    direction_signal: np.ndarray,
    stop_loss: float | None,
    take_profit: float | None,
    size: np.ndarray | None = None,
) -> dict:
    signal = np.where(direction_signal > 0, 1, -1)
    return simulate_trades(
        entry,
        exit_,
        signal,
        stop_loss=stop_loss,
        take_profit=take_profit,
        size=size,
    )


def run_backtest(
    dataset_path: Path,
    close_index: int,
    epochs: int,
    batch_size: int,
    stop_loss: float | None,
    take_profit: float | None,
    min_confidence: float,
    dynamic_sizing: bool,
) -> dict:
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
    rf_metrics = backtest_model(
        entry_prices,
        true_prices,
        rf_signal,
        stop_loss,
        take_profit,
    )

    trainer = HybridTrainer(dataset, config=HybridConfig())
    trainer.train(epochs=epochs, batch_size=batch_size)
    dir_pred, level_pred = trainer.model.model.predict(dataset.test_X, verbose=0)
    probs = dir_pred.flatten()

    hybrid_signal = np.zeros_like(probs)
    if min_confidence <= 0:
        hybrid_signal = np.sign(level_pred.flatten() - entry_prices)
    else:
        lower = 1.0 - min_confidence
        for idx, p in enumerate(probs):
            if p >= min_confidence:
                hybrid_signal[idx] = 1
            elif p <= lower:
                hybrid_signal[idx] = -1
            else:
                hybrid_signal[idx] = 0

    size = None
    if dynamic_sizing:
        size = np.clip(np.abs(probs - 0.5) * 2, 0.0, 1.0)

    hybrid_metrics = backtest_model(
        entry_prices,
        true_prices,
        hybrid_signal,
        stop_loss,
        take_profit,
        size=size,
    )

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
        results = run_backtest(
            path,
            args.close_index,
            args.epochs,
            args.batch_size,
            args.stop_loss,
            args.take_profit,
            args.min_confidence,
            args.dynamic_sizing,
        )
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

"""Simple hyperparameter sweep for the hybrid model."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

import yaml
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.hybrid_model import HybridConfig  # noqa: E402
from models.trainer import HybridTrainer, SequenceDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for the hybrid model.")
    parser.add_argument("dataset", help="Path to .npz dataset.")
    parser.add_argument("--config", default="config/config.yaml", help="Config file with tuning grid.")
    parser.add_argument("--close-index", type=int, default=-1, help="Index of close price.")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per trial.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per trial.")
    parser.add_argument("--output", default="data/processed/sequences/hybrid_tuning.json", help="Where to store metrics.")
    return parser.parse_args()


def load_tuning_grid(path: str) -> Dict[str, List]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("model", {}).get("tuning", {})


def main() -> None:
    args = parse_args()
    base_dataset = SequenceDataset.from_npz(args.dataset, close_index=args.close_index)
    grid = load_tuning_grid(args.config)
    dropout_values = grid.get("dropout", [0.3])
    head_values = grid.get("num_heads", [4])
    seq_lengths = grid.get("sequence_length", [base_dataset.train_X.shape[1]])

    results = []
    for dropout, heads, seq_len in product(dropout_values, head_values, seq_lengths):
        print(f"Running trial dropout={dropout}, heads={heads}, seq_len={seq_len}")
        dataset = (
            _clone_dataset(base_dataset)
            if seq_len == base_dataset.train_X.shape[1]
            else _adjust_sequence_length(base_dataset, seq_len)
        )
        config = HybridConfig(
            dropout=dropout,
            num_heads=heads,
            transformer_dim=dataset.train_X.shape[2],
            transformer_ff=dataset.train_X.shape[2] * 2,
        )
        trainer = HybridTrainer(dataset, config=config)
        trainer.train(epochs=args.epochs, batch_size=args.batch_size)
        metrics = trainer.evaluate()
        results.append(
            {
                "dropout": dropout,
                "num_heads": heads,
                "sequence_length": seq_len,
                "metrics": metrics,
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Tuning results saved to {args.output}")


def _resize_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    if X.shape[1] == seq_len:
        return X.copy()
    if X.shape[1] > seq_len:
        return X[:, -seq_len:, :]
    pad_width = seq_len - X.shape[1]
    pad = np.repeat(X[:, :1, :], pad_width, axis=1)
    return np.concatenate([pad, X], axis=1)


def _clone_dataset(dataset: SequenceDataset) -> SequenceDataset:
    return SequenceDataset(
        train_X=dataset.train_X.copy(),
        train_y=dataset.train_y.copy(),
        val_X=dataset.val_X.copy(),
        val_y=dataset.val_y.copy(),
        test_X=dataset.test_X.copy(),
        test_y=dataset.test_y.copy(),
        feature_names=dataset.feature_names,
        close_index=dataset.close_index,
    )


def _adjust_sequence_length(base_dataset: SequenceDataset, seq_len: int) -> SequenceDataset:
    cloned = _clone_dataset(base_dataset)
    cloned.train_X = _resize_sequences(cloned.train_X, seq_len)
    cloned.val_X = _resize_sequences(cloned.val_X, seq_len)
    cloned.test_X = _resize_sequences(cloned.test_X, seq_len)
    return cloned


if __name__ == "__main__":
    main()


"""CLI for training the hybrid probabilistic model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.hybrid_model import HybridConfig  # noqa: E402
from models.trainer import HybridTrainer, SequenceDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid LSTM+Transformer model.")
    parser.add_argument("dataset", help="Path to .npz sequence dataset.")
    parser.add_argument("--config", default="config/config.yaml", help="YAML config for model params.")
    parser.add_argument("--close-index", type=int, default=-1, help="Index of close price feature.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--output", help="Optional path to save evaluation metrics JSON.")
    return parser.parse_args()


def load_model_config(path: str) -> HybridConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return HybridConfig()
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    model_cfg = data.get("model", {})
    probabilistic_cfg = data.get("model", {}).get("probabilistic", {})
    return HybridConfig(
        lstm_units=tuple(model_cfg.get("lstm", {}).get("hidden_units", [128, 64])),
        dropout=model_cfg.get("lstm", {}).get("dropout", 0.3),
        num_heads=model_cfg.get("transformer", {}).get("num_heads", 8),
        transformer_dim=model_cfg.get("transformer", {}).get("d_model", 128),
        transformer_ff=model_cfg.get("transformer", {}).get("d_ff", 256),
        learning_rate=model_cfg.get("training", {}).get("learning_rate", 1e-3),
        monte_carlo_samples=probabilistic_cfg.get("monte_carlo_samples", 50),
    )


def main() -> None:
    args = parse_args()
    dataset = SequenceDataset.from_npz(args.dataset, close_index=args.close_index)
    config = load_model_config(args.config)
    trainer = HybridTrainer(dataset, config=config)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


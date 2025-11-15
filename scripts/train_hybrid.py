"""CLI for training the hybrid probabilistic model with best practices."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
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
    parser.add_argument("--history-output", help="Optional path to save the Keras history JSON.")
    parser.add_argument("--save-model", default="models/hybrid_model.keras", help="Where to save the trained model.")
    parser.add_argument("--no-save-model", action="store_true", help="Skip saving the final model.")
    parser.add_argument(
        "--checkpoint-dir",
        default="models/checkpoints",
        help="Directory to store training checkpoints.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable checkpointing even if a directory is provided.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        help="Optional directory for TensorBoard logs. Disabled if not provided.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Early stopping patience (set 0 or negative to disable).",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training for supported hardware.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
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


def set_random_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_mixed_precision(enabled: bool) -> None:
    if not enabled:
        return
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")


def build_callbacks(
    args: argparse.Namespace,
    dataset_label: str,
) -> tuple[list[tf.keras.callbacks.Callback], Path | None]:
    callbacks: list[tf.keras.callbacks.Callback] = []
    checkpoint_path: Path | None = None

    if args.early_stop_patience and args.early_stop_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.early_stop_patience,
                restore_best_weights=True,
            )
        )

    if not args.no_checkpoints:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / f"{dataset_label}_{datetime.utcnow():%Y%m%d%H%M%S}.keras"
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
            )
        )

    if args.tensorboard_dir:
        log_dir = Path(args.tensorboard_dir) / dataset_label / datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
            )
        )

    return callbacks, checkpoint_path


def ensure_path_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{description} not found: {path}")
    return path


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = ensure_path_exists(Path(args.dataset), "Dataset file")
    set_random_seeds(args.seed)
    configure_mixed_precision(args.mixed_precision)

    dataset = SequenceDataset.from_npz(dataset_path, close_index=args.close_index)
    config = load_model_config(args.config)
    trainer = HybridTrainer(dataset, config=config)

    dataset_label = dataset_path.stem
    callbacks, checkpoint_path = build_callbacks(args, dataset_label)
    history = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    if args.output:
        dump_json(Path(args.output), metrics)
    if args.history_output:
        dump_json(Path(args.history_output), history.history)
    if not args.no_save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.model.model.save(save_path)
        print(f"Saved trained model to {save_path}")
    if checkpoint_path is not None:
        print(f"Best checkpoint stored at {checkpoint_path}")


if __name__ == "__main__":
    main()
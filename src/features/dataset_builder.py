"""Utilities for assembling supervised learning datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import get_settings
from utils.logger import setup_logger

logger = setup_logger("dataset_builder")


@dataclass
class SplitConfig:
    train: float = 0.7
    validation: float = 0.2
    test: float = 0.1

    def as_indices(self, total: int) -> Tuple[int, int]:
        train_end = int(total * self.train)
        val_end = train_end + int(total * self.validation)
        return train_end, val_end


def generate_sequences(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    sequence_length: int = 60,
) -> Tuple[np.ndarray, np.ndarray, Sequence[str]]:
    """Transform a dataframe into rolling window sequences."""
    feature_matrix = df[feature_columns].values
    target_vector = df[target_column].values

    sequences = []
    labels = []
    for idx in range(sequence_length, len(df)):
        sequences.append(feature_matrix[idx - sequence_length : idx])
        labels.append(target_vector[idx])

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    logger.info(
        "Generated supervised sequences",
        extra={"sequence_length": sequence_length, "samples": len(X)},
    )
    return X, y, list(feature_columns)


def split_sequences(
    X: np.ndarray, y: np.ndarray, config: SplitConfig
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    total = len(X)
    train_end, val_end = config.as_indices(total)
    datasets = {
        "train": (X[:train_end], y[:train_end]),
        "validation": (X[train_end:val_end], y[train_end:val_end]),
        "test": (X[val_end:], y[val_end:]),
    }
    logger.info(
        "Split sequences",
        extra={k: len(v[0]) for k, v in datasets.items()},
    )
    return datasets


def save_sequence_pack(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    basename: str,
    feature_names: Sequence[str] | None = None,
) -> Path:
    """Persist train/val/test splits as compressed numpy archives."""
    settings = get_settings()
    out_dir = settings.data_processed_dir / "sequences"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{basename}.npz"
    save_kwargs = {
        "train_X": datasets["train"][0],
        "train_y": datasets["train"][1],
        "val_X": datasets["validation"][0],
        "val_y": datasets["validation"][1],
        "test_X": datasets["test"][0],
        "test_y": datasets["test"][1],
    }
    if feature_names is not None:
        save_kwargs["feature_names"] = np.array(feature_names)
    np.savez_compressed(path, **save_kwargs)
    logger.info("Saved sequence pack", extra={"path": str(path)})
    return path


def build_feature_frame(price_df: pd.DataFrame, *feature_frames: pd.DataFrame) -> pd.DataFrame:
    combined = price_df.copy()
    for frame in feature_frames:
        combined = combined.join(frame, how="left")
    combined = combined.dropna()
    return combined


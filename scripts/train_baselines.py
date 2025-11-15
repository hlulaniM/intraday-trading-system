"""CLI for training baseline models on saved sequence datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.trainer import SequenceDataset, train_all_baselines  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline models on a sequence dataset.")
    parser.add_argument("dataset", help="Path to .npz file created by save_sequence_pack.")
    parser.add_argument("--close-index", type=int, default=-1, help="Index of the close price feature.")
    parser.add_argument("--output", help="Optional path to write metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SequenceDataset.from_npz(args.dataset, close_index=args.close_index)
    results = train_all_baselines(dataset)
    summary = {name: result.metrics for name, result in results.items()}
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


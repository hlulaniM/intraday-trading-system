"""Feature engineering modules."""

from .dataset_builder import (
    SplitConfig,
    build_feature_frame,
    generate_sequences,
    save_sequence_pack,
    split_sequences,
)
from .sentiment import SentimentRecord, merge_sentiment, sentiment_frame
from .technical import add_indicators

__all__ = [
    "add_indicators",
    "SentimentRecord",
    "sentiment_frame",
    "merge_sentiment",
    "SplitConfig",
    "generate_sequences",
    "split_sequences",
    "save_sequence_pack",
    "build_feature_frame",
]


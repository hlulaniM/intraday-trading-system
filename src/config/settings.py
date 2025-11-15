"""Application settings loader with environment variable support."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Resolve project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv_files() -> None:
    """
    Load `.env` style files if they exist.

    We attempt multiple locations so developers can choose their preferred workflow
    (e.g., `.env`, `.env.local`, `config/.secrets.env`). Missing files are ignored.
    """
    candidate_files = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / ".env.local",
        PROJECT_ROOT / "config" / ".env",
        PROJECT_ROOT / "config" / ".env.local",
    ]

    for env_file in candidate_files:
        if env_file.exists():
            load_dotenv(env_file, override=False)


_load_dotenv_files()


@dataclass(frozen=True)
class Settings:
    """Container for application-wide configuration values."""

    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    logs_dir: Path = PROJECT_ROOT / "logs"

    @staticmethod
    def _get_env(name: str, default: Optional[str] = None) -> str:
        value = os.getenv(name, default)
        if value is None:
            raise RuntimeError(f"Environment variable '{name}' is required but missing.")
        return value

    @classmethod
    def load(cls) -> "Settings":
        """Instantiate settings from environment variables."""
        return cls(
            alpaca_api_key=cls._get_env("ALPACA_API_KEY"),
            alpaca_secret_key=cls._get_env("ALPACA_SECRET_KEY"),
            alpaca_base_url=cls._get_env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"),
            data_raw_dir=Path(os.getenv("DATA_RAW_DIR", PROJECT_ROOT / "data" / "raw")),
            data_processed_dir=Path(
                os.getenv("DATA_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed")
            ),
            logs_dir=Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings.load()


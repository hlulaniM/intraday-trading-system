"""Utility helpers for normalizing symbol identifiers."""

from __future__ import annotations


def sanitize_symbol(symbol: str) -> str:
    """Return an uppercase filesystem-safe symbol."""
    return symbol.upper().replace("/", "_").replace(" ", "_")


def symbol_slug(symbol: str) -> str:
    """Return a lowercase slug suitable for filenames."""
    return sanitize_symbol(symbol).lower()


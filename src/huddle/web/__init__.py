"""The model-manager web page: static files only, no server-rendered templates."""

from __future__ import annotations

from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

__all__ = ["STATIC_DIR"]

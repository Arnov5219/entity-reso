"""Utility helpers for business entity resolution."""
import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "dataset").exists() or (parent / "src").exists():
            return parent
    return current.parent.parent.parent

"""Backward-compatibility module: re-exports EntityMatcher from gbdt."""
from .gbdt import EntityMatcher, GBDTMatcher

__all__ = ["EntityMatcher", "GBDTMatcher"]

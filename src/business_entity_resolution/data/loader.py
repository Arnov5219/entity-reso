"""Data loading and dataset management module for Business Entity Resolution."""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd


class DataLoader:
    """Loads source TSV files and ground truth for Business Entity Resolution."""

    def __init__(self, data_dir: str = "dataset", split: str = "train", cache: bool = False):
        self.data_dir = Path(data_dir)
        # Handle relative path from notebooks/ or scripts/
        if not self.data_dir.exists() and (Path("..") / data_dir).exists():
            self.data_dir = Path("..") / data_dir
        self.split = split
        self.split_dir = self.data_dir / split
        self.cache_enabled = cache
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_source(self, source_num: int, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load a specific source file (source1, source2, or source3).
        Preserves string data types, prevents NaN conversion, and uses explicit tab separators.
        """
        cache_key = f"{self.split}_source{source_num}_{nrows}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy()

        filename = f"{self.split}_source{source_num}.tsv"
        filepath = self.split_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Source file not found: {filepath}")

        df = pd.read_csv(
            filepath,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            nrows=nrows,
            encoding="utf-8"
        )
        if self.cache_enabled:
            self._cache[cache_key] = df
        return df

    def load_ground_truth(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load ground truth file (available for train split).
        Preserves exact tab separation, string dtypes, and empty strings for unmatched entities.
        """
        cache_key = f"{self.split}_ground_truth_{nrows}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy()

        filename = f"{self.split}_ground_truth.tsv"
        filepath = self.split_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Ground truth file not found: {filepath}")

        df = pd.read_csv(
            filepath,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            nrows=nrows,
            encoding="utf-8"
        )
        if self.cache_enabled:
            self._cache[cache_key] = df
        return df

    def load_all(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three sources (source1, source2, source3)."""
        s1 = self.load_source(1, nrows=nrows)
        s2 = self.load_source(2, nrows=nrows)
        s3 = self.load_source(3, nrows=nrows)
        return s1, s2, s3

    def load_all_with_truth(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three sources and ground truth."""
        s1, s2, s3 = self.load_all(nrows=nrows)
        gt = self.load_ground_truth(nrows=nrows)
        return s1, s2, s3, gt

    def get_train_val_split(
        self,
        total_sample: int = 10000,
        val_size: int = 2000,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reproducible entity-level train/validation split without data leakage.

        Returns:
            train_gt: DataFrame of training ground truth
            val_gt: DataFrame of validation ground truth
        """
        gt_full = self.load_ground_truth()
        sample = gt_full.sample(n=min(total_sample, len(gt_full)), random_state=random_state).reset_index(drop=True)
        train_count = len(sample) - val_size
        train_gt = sample.iloc[:train_count].reset_index(drop=True)
        val_gt = sample.iloc[train_count:].reset_index(drop=True)
        return train_gt, val_gt

    def clear_cache(self):
        """Clear memory cache of loaded DataFrames."""
        self._cache.clear()

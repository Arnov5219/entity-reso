import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


class DataLoader:
    """Loads source TSV files and ground truth for Business Entity Resolution."""

    def __init__(self, data_dir: str = "dataset", split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split

    def load_source(self, source_num: int, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load a specific source file (source1, source2, or source3)."""
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
        return df

    def load_ground_truth(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load ground truth file (only available for train split)."""
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
        return df

    def load_all(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three sources."""
        s1 = self.load_source(1, nrows=nrows)
        s2 = self.load_source(2, nrows=nrows)
        s3 = self.load_source(3, nrows=nrows)
        return s1, s2, s3

    def load_all_with_truth(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three sources and ground truth."""
        s1, s2, s3 = self.load_all(nrows=nrows)
        gt = self.load_ground_truth(nrows=nrows)
        return s1, s2, s3, gt

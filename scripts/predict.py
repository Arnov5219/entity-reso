#!/usr/bin/env python3
"""Thin CLI entry point to run inference using trained Entity Resolution model."""
import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from business_entity_resolution.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Entity Resolution inference")
    parser.add_argument("--dataset-dir", default="dataset", help="Path to dataset directory")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--model-path", default="experiments/results/model.joblib", help="Path to trained model")
    parser.add_argument("--split", default="test", help="Data split to evaluate (test or train)")
    parser.add_argument("--max-records", type=int, default=None, help="Max S1 records (None for all)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Decision threshold")
    args = parser.parse_args()

    pipe = EntityResolutionPipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_type="gbdt",
        threshold=args.threshold
    )

    pipe.run(
        train_split="train",
        test_split=args.split,
        save_model_path=args.model_path,
        max_test_records=args.max_records
    )


if __name__ == "__main__":
    main()

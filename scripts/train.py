#!/usr/bin/env python3
"""Thin CLI entry point to train the Entity Resolution model."""
import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from business_entity_resolution.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(description="Train Entity Resolution GBDT model")
    parser.add_argument("--dataset-dir", default="dataset", help="Path to dataset directory")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--model-out", default="experiments/results/model.joblib", help="Path to save trained model")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of S1 entities to train on")
    parser.add_argument("--threshold", type=float, default=0.70, help="Decision threshold")
    args = parser.parse_args()

    pipe = EntityResolutionPipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_type="gbdt",
        threshold=args.threshold
    )

    print(f"Training model on {args.sample_size} S1 samples...")
    pipe.train(train_split="train", sample_size=args.sample_size, random_state=42)
    pipe.matcher.save(args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()

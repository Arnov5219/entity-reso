#!/usr/bin/env python3
"""CLI script to run reproducible model evaluation for Business Entity Resolution."""
import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from business_entity_resolution.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution GBDT model on validation split")
    parser.add_argument("--dataset-dir", default="dataset", help="Path to dataset directory")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--model-path", default="experiments/results/model.joblib", help="Path to trained model")
    parser.add_argument("--baseline-path", default="experiments/results/metrics.json", help="Path to Phase 2 baseline metrics")
    parser.add_argument("--output-metrics", default="output/evaluation_metrics.json", help="Path to save evaluation metrics")
    parser.add_argument("--val-size", type=int, default=2000, help="Number of validation S1 entities")
    parser.add_argument("--val-sample-size", type=int, default=10000, help="Total sample size for train/val split")
    parser.add_argument("--threshold", type=float, default=0.70, help="Decision threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split")
    args = parser.parse_args()

    pipe = EntityResolutionPipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_type="gbdt",
        threshold=args.threshold
    )

    metrics = pipe.evaluate(
        val_sample_size=args.val_sample_size,
        val_size=args.val_size,
        random_state=args.seed,
        model_path=args.model_path,
        baseline_path=args.baseline_path,
        save_metrics_path=args.output_metrics
    )

    if not metrics:
        print("Evaluation failed: no metrics generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()

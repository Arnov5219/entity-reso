#!/usr/bin/env python3
"""Unified Reproducible Machine Learning Pipeline for Business Entity Resolution.

Supports three operational modes:
  - train:   Train GBDT model on training data, evaluate on held-out validation data, save model and metrics.
  - predict: Load existing trained model, run inference on test data, generate and validate submission files.
  - full:    Execute complete end-to-end workflow (Train -> Validate -> Predict -> Validate Submission).

Works both locally and in cloud environments like Google Colab.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure src/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from business_entity_resolution.pipeline import EntityResolutionPipeline
from business_entity_resolution.data.loader import DataLoader
from business_entity_resolution.blocking.blocker import EntityBlocker
from business_entity_resolution.features.feature_engineering import FeatureExtractor
from business_entity_resolution.models.gbdt import EntityMatcher
from business_entity_resolution.evaluation.metrics import evaluate_predictions, save_metrics, compare_to_baseline
from utils.validate_submission import validate as validate_submission_files


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configure structured console logging."""
    logger = logging.getLogger("EntityResolutionPipeline")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from JSON file if provided."""
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def run_train_mode(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    """Mode A: Train GBDT model and evaluate on held-out validation data."""
    logger.info("=" * 70)
    logger.info("MODE: TRAIN — Starting Model Training & Validation Workflow")
    logger.info("=" * 70)
    logger.info(f"Dataset directory:       {args.dataset_dir}")
    logger.info(f"Model output path:       {args.model_path}")
    logger.info(f"Metrics output path:     {args.metrics_path}")
    logger.info(f"Sample size:             {args.sample_size:,} total ({args.sample_size - args.val_size:,} train, {args.val_size:,} val)")
    logger.info(f"Random seed:             {args.seed}")
    logger.info(f"Threshold:               {args.threshold}")

    # Verify dataset exists
    train_dir = Path(args.dataset_dir) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training dataset directory not found at: {train_dir}. "
            "Please check --dataset-dir (e.g., Google Drive mount path in Colab)."
        )

    t0 = time.time()
    pipe = EntityResolutionPipeline(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        model_type="gbdt",
        threshold=args.threshold
    )

    # 1. Fit model on training split
    train_count = args.sample_size - args.val_size
    logger.info(f"Fitting GBDT classifier on {train_count:,} S1 entities...")
    pipe.train(
        train_split="train",
        sample_size=train_count,
        random_state=args.seed
    )

    # 2. Save trained model
    model_out = Path(args.model_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    pipe.matcher.save(model_out)
    logger.info(f"Successfully saved trained model to: {model_out}")

    # 3. Evaluate on strictly held-out validation split
    logger.info(f"Evaluating model on {args.val_size:,} held-out validation entities...")
    metrics = pipe.evaluate(
        val_sample_size=args.sample_size,
        val_size=args.val_size,
        random_state=args.seed,
        model_path=str(model_out),
        baseline_path=str(args.baseline_path),
        save_metrics_path=str(args.metrics_path)
    )

    elapsed = time.time() - t0
    logger.info(f"Training and validation completed in {elapsed:.1f}s.")
    logger.info(f"Primary Validation Macro F0.5: {metrics.get('macro_f05', 0.0):.4f}")
    return metrics


def run_predict_mode(args: argparse.Namespace, logger: logging.Logger) -> Tuple[Path, Path]:
    """Mode B: Load model and run inference to generate submission files."""
    logger.info("=" * 70)
    logger.info("MODE: PREDICT — Starting Test Inference & Submission Generation")
    logger.info("=" * 70)
    logger.info(f"Dataset directory:       {args.dataset_dir}")
    logger.info(f"Model path:              {args.model_path}")
    logger.info(f"Output directory:        {args.output_dir}")
    logger.info(f"Decision threshold:      {args.threshold}")
    if args.max_test_records:
        logger.info(f"Max test records:        {args.max_test_records:,} (smoke test)")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {model_path}. "
            "Please run with `--mode train` first or specify a valid --model-path."
        )

    test_dir = Path(args.dataset_dir) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test dataset directory not found at: {test_dir}. "
            "Please verify --dataset-dir points to the folder containing 'train/' and 'test/'."
        )

    t0 = time.time()
    pipe = EntityResolutionPipeline(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        model_type="gbdt",
        threshold=args.threshold
    )

    logger.info(f"Running inference across test partitions using model: {model_path}...")
    pipe.run(
        train_split="train",
        test_split="test",
        save_model_path=str(model_path),
        max_test_records=args.max_test_records
    )

    match_file = Path(args.output_dir) / "matching_results.tsv"
    cand_file = Path(args.output_dir) / "candidate_pairs.tsv"

    if not match_file.exists():
        raise RuntimeError(f"Expected output file missing: {match_file}")
    if not cand_file.exists():
        raise RuntimeError(f"Expected output file missing: {cand_file}")

    elapsed = time.time() - t0
    logger.info(f"Inference completed in {elapsed:.1f}s.")
    logger.info(f"Generated matching results: {match_file} ({match_file.stat().st_size:,} bytes)")
    logger.info(f"Generated candidate pairs:  {cand_file} ({cand_file.stat().st_size:,} bytes)")

    # 4. Submission validation
    if not args.skip_validation:
        if args.max_test_records:
            logger.info("Smoke test detected (--max-test-records set): Skipping full 1.7M-row submission check.")
            logger.info(f"Verified {match_file.name} ({match_file.stat().st_size:,} bytes) and {cand_file.name} ({cand_file.stat().st_size:,} bytes) written successfully.")
        else:
            logger.info("\nValidating generated submission against competition rules...")
            errors, warnings = validate_submission_files(
                matching_path=str(match_file),
                candidate_path=str(cand_file),
                test_dir=str(test_dir),
                check_ids=False
            )
            for w in warnings:
                logger.warning(f"Submission Warning: {w}")
            if errors:
                logger.error(f"Submission Validation FAILED with {len(errors)} error(s):")
                for i, err in enumerate(errors, 1):
                    logger.error(f"  {i}. {err}")
                raise ValueError("Generated submission failed validation checks!")
            else:
                logger.info("Submission Validation PASSED: Files are safe for leaderboard submission.")

    return match_file, cand_file


def run_full_pipeline(args: argparse.Namespace, logger: logging.Logger):
    """Mode C: Full end-to-end execution (Train -> Validate -> Predict -> Validate Submission)."""
    logger.info("=" * 70)
    logger.info("MODE: FULL — Executing Complete End-to-End Pipeline")
    logger.info("=" * 70)
    t_start = time.time()

    # Step 1: Train & Validate
    metrics = run_train_mode(args, logger)

    # Step 2: Predict & Validate Submission
    match_file, cand_file = run_predict_mode(args, logger)

    total_time = time.time() - t_start
    logger.info("=" * 70)
    logger.info("FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"Total pipeline runtime:      {total_time:.1f}s ({total_time / 60:.2f} min)")
    logger.info(f"Validation Macro F0.5:       {metrics.get('macro_f05', 0.0):.4f}")
    logger.info(f"Final Model Saved:           {args.model_path}")
    logger.info(f"Final Metrics Saved:         {args.metrics_path}")
    logger.info(f"Official Submission File:    {match_file}")
    logger.info(f"Candidate Pairs File:        {cand_file}")
    logger.info("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser with sensible defaults and config overrides."""
    parser = argparse.ArgumentParser(
        description="Unified End-to-End Pipeline for Business Entity Resolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "full"],
        default="full",
        help="Pipeline operational mode: 'train', 'predict', or 'full'"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to JSON configuration file to override defaults"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Path to dataset root folder containing 'train/' and 'test/'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Path to folder where predictions and candidate files will be written"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="experiments/results/model.joblib",
        help="Path where trained model will be saved or loaded from"
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="output/evaluation_metrics.json",
        help="Path to save evaluation metrics (.json and .csv)"
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="experiments/results/metrics.json",
        help="Path to Phase 2 baseline metrics for comparison"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Total sample size of S1 entities for train/validation split"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=2000,
        help="Number of held-out S1 validation entities"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Decision probability threshold for declaring matches"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset splits and model training"
    )
    parser.add_argument(
        "--max-test-records",
        type=int,
        default=None,
        help="Optional limit on test S1 records for quick smoke tests"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip submission format validation"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity"
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Apply configuration file overrides if provided
    if args.config:
        cfg = load_config(args.config)
        for key, val in cfg.items():
            attr_key = key.replace("-", "_")
            if hasattr(args, attr_key):
                setattr(args, attr_key, val)

    logger = setup_logger(args.log_level)

    try:
        if args.mode == "train":
            run_train_mode(args, logger)
        elif args.mode == "predict":
            run_predict_mode(args, logger)
        elif args.mode == "full":
            run_full_pipeline(args, logger)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
    except Exception as exc:
        logger.error(f"Pipeline failed with exception: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

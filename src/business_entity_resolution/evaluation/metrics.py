"""Model evaluation metrics module for Business Entity Resolution.

Implements official hackathon F0.5 macro evaluation, pairwise metrics,
confusion matrix counts (TP, FP, FN, TN), baseline comparison, and metric serialization.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd


def compute_entity_metrics(true_set: Set[str], pred_set: Set[str]) -> Dict[str, Union[float, int]]:
    """
    Compute precision, recall, f0.5, f1, and exact match accuracy for a single S1 entity.

    Follows the official hackathon scoring formula:
    - F0.5 = (1.25 * P * R) / (0.25 * P + R)
    - Entities with no true matches and no predictions receive 1.0 on all metrics.
    """
    if not true_set and not pred_set:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f05": 1.0,
            "f1": 1.0,
            "accuracy": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    if not true_set and pred_set:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f05": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "tp": 0,
            "fp": len(pred_set),
            "fn": 0,
        }
    if true_set and not pred_set:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f05": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(true_set),
        }

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f05": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "tp": 0,
            "fp": fp,
            "fn": fn,
        }

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = 1.0 if true_set == pred_set else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_predictions(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    candidates_df: Optional[pd.DataFrame] = None,
    save_path: Optional[Union[str, Path]] = None,
    baseline_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth according to hackathon F0.5 evaluation metric.

    Parameters:
        ground_truth_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
        predictions_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
        candidates_df: Optional DataFrame with ['source1_entity_id', 'candidate_entity_ids']
                       Used to compute candidate-level True Negatives (TN) and pairwise accuracy.
        save_path: Optional path (.json or .csv) to save evaluation metrics.
        baseline_path: Optional path to baseline metrics.json to compute delta comparison.

    Returns:
        Dict containing all macro metrics, pairwise metrics, and confusion counts.
    """
    # 1. Parse ground truth map
    gt_map: Dict[str, Set[str]] = {}
    for _, row in ground_truth_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        val = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        ids = set(x.strip() for x in val.split(",") if x.strip()) if val else set()
        gt_map[s1] = ids

    # 2. Parse predictions map
    pred_map: Dict[str, Set[str]] = {}
    for _, row in predictions_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        val = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        ids = set(x.strip() for x in val.split(",") if x.strip()) if val else set()
        pred_map[s1] = ids

    # 3. Optional parse candidate map for TN calculation
    cand_map: Dict[str, Set[str]] = {}
    if candidates_df is not None:
        cand_col = "candidate_entity_ids" if "candidate_entity_ids" in candidates_df.columns else "matched_entity_ids"
        for _, row in candidates_df.iterrows():
            s1 = str(row["source1_entity_id"]).strip()
            val = str(row[cand_col]).strip() if pd.notna(row[cand_col]) else ""
            ids = set(x.strip() for x in val.split(",") if x.strip()) if val else set()
            cand_map[s1] = ids

    all_s1 = sorted(set(gt_map.keys()))
    if not all_s1:
        return {}

    precisions: List[float] = []
    recalls: List[float] = []
    f05s: List[float] = []
    f1s: List[float] = []
    accuracies: List[float] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_preds = 0
    total_candidates = 0

    for s1 in all_s1:
        true_set = gt_map.get(s1, set())
        pred_set = pred_map.get(s1, set())
        total_preds += len(pred_set)

        res = compute_entity_metrics(true_set, pred_set)
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        f05s.append(res["f05"])
        f1s.append(res["f1"])
        accuracies.append(res["accuracy"])
        total_tp += int(res["tp"])
        total_fp += int(res["fp"])
        total_fn += int(res["fn"])

        if cand_map:
            cands = cand_map.get(s1, set())
            total_candidates += len(cands)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f05 = float(np.mean(f05s))
    macro_f1 = float(np.mean(f1s))
    exact_accuracy = float(np.mean(accuracies))

    # Pairwise aggregates
    pairwise_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    pairwise_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    denom_f05 = (0.25 * pairwise_precision + pairwise_recall)
    pairwise_f05 = (1.25 * pairwise_precision * pairwise_recall) / denom_f05 if denom_f05 > 0 else 0.0
    denom_f1 = (pairwise_precision + pairwise_recall)
    pairwise_f1 = (2.0 * pairwise_precision * pairwise_recall) / denom_f1 if denom_f1 > 0 else 0.0

    # True negatives and pairwise accuracy
    total_tn = None
    pairwise_accuracy = None
    if total_candidates > 0:
        total_tn = total_candidates - total_tp - total_fp
        pairwise_accuracy = (total_tp + total_tn) / total_candidates

    metrics: Dict[str, Any] = {
        "macro_f05": macro_f05,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "exact_accuracy": exact_accuracy,
        "total_s1_entities": len(all_s1),
        "total_predictions": total_preds,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "pairwise_precision": pairwise_precision,
        "pairwise_recall": pairwise_recall,
        "pairwise_f05": pairwise_f05,
        "pairwise_f1": pairwise_f1,
    }

    if total_tn is not None:
        metrics["total_tn"] = total_tn
    if pairwise_accuracy is not None:
        metrics["pairwise_accuracy"] = pairwise_accuracy

    # Save to file if requested
    if save_path:
        save_metrics(metrics, save_path)

    # Compare with baseline if provided
    if baseline_path:
        baseline_file = Path(baseline_path)
        if not baseline_file.exists() and (Path("..") / baseline_path).exists():
            baseline_file = Path("..") / baseline_path
        if baseline_file.exists():
            metrics["baseline_comparison"] = compare_to_baseline(metrics, baseline_file)

    return metrics


def save_metrics(metrics: Dict[str, Any], filepath: Union[str, Path]):
    """Save metrics dictionary to JSON or CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten baseline comparison if saving to CSV
    metrics_to_save = dict(metrics)

    if path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics_to_save, f, indent=2)
    elif path.suffix.lower() == ".csv":
        flat_records = []
        for k, v in metrics_to_save.items():
            if k == "baseline_comparison":
                continue
            flat_records.append({"metric": k, "value": v})
        pd.DataFrame(flat_records).to_csv(path, index=False)
    else:
        # Default to json
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(metrics_to_save, f, indent=2)


def compare_to_baseline(
    current_metrics: Dict[str, Any],
    baseline: Union[Dict[str, Any], str, Path]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare current evaluation metrics with baseline metrics.

    Returns:
        Dict mapping metric_name -> {'current': val, 'baseline': val, 'diff': val, 'match': bool}
    """
    if isinstance(baseline, (str, Path)):
        p = Path(baseline)
        if not p.exists() and (Path("..") / baseline).exists():
            p = Path("..") / baseline
        with open(p, encoding="utf-8") as f:
            baseline_data = json.load(f)
    else:
        baseline_data = baseline

    comparison: Dict[str, Dict[str, Any]] = {}
    for k, base_val in baseline_data.items():
        if k in current_metrics:
            curr_val = current_metrics[k]
            if isinstance(base_val, (int, float)) and isinstance(curr_val, (int, float)):
                diff = float(curr_val) - float(base_val)
                is_match = abs(diff) < 1e-4
                comparison[k] = {
                    "current": curr_val,
                    "baseline": base_val,
                    "diff": diff,
                    "match": is_match
                }
    return comparison

from typing import Dict, Any, Union, Set
import numpy as np
import pandas as pd


def compute_entity_metrics(true_set: Set[str], pred_set: Set[str]) -> Dict[str, float]:
    """Compute precision, recall, f0.5, f1, and exact match accuracy for a single S1 entity."""
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
    predictions_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth according to hackathon F0.5 evaluation metric.

    ground_truth_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
    predictions_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
    """
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        val = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        ids = set(x.strip() for x in val.split(",") if x.strip()) if val else set()
        gt_map[s1] = ids

    pred_map = {}
    for _, row in predictions_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        val = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        ids = set(x.strip() for x in val.split(",") if x.strip()) if val else set()
        pred_map[s1] = ids

    all_s1 = sorted(set(gt_map.keys()))
    if not all_s1:
        return {}

    precisions = []
    recalls = []
    f05s = []
    f1s = []
    accuracies = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_preds = 0

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
        total_tp += res["tp"]
        total_fp += res["fp"]
        total_fn += res["fn"]

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f05 = float(np.mean(f05s))
    macro_f1 = float(np.mean(f1s))
    exact_accuracy = float(np.mean(accuracies))

    return {
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
    }

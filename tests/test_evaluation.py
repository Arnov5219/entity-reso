import tempfile
import unittest
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from business_entity_resolution.evaluation.metrics import (
    compute_entity_metrics,
    evaluate_predictions,
    save_metrics,
    compare_to_baseline
)


class TestEvaluation(unittest.TestCase):
    def test_compute_entity_metrics_perfect_match(self):
        m = compute_entity_metrics({"S2-001", "S3-002"}, {"S2-001", "S3-002"})
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f05"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)

    def test_compute_entity_metrics_empty_match(self):
        m = compute_entity_metrics(set(), set())
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f05"], 1.0)
        self.assertEqual(m["accuracy"], 1.0)

    def test_compute_entity_metrics_partial(self):
        # 1 TP, 1 FP (pred S2-002 not in true), 1 FN (true S2-003 not in pred)
        m = compute_entity_metrics({"S2-001", "S2-003"}, {"S2-001", "S2-002"})
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 1)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["f05"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)
        self.assertEqual(m["accuracy"], 0.0)

    def test_evaluate_predictions_dataframe(self):
        gt_df = pd.DataFrame([
            {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-001,S3-001"},
            {"source1_entity_id": "S1-002", "matched_entity_ids": ""},
        ])
        pred_df = pd.DataFrame([
            {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-001"},
            {"source1_entity_id": "S1-002", "matched_entity_ids": ""},
        ])
        cand_df = pd.DataFrame([
            {"source1_entity_id": "S1-001", "candidate_entity_ids": "S2-001,S3-001,S2-999"},
            {"source1_entity_id": "S1-002", "candidate_entity_ids": "S3-888"},
        ])

        metrics = evaluate_predictions(gt_df, pred_df, candidates_df=cand_df)
        self.assertEqual(metrics["total_s1_entities"], 2)
        self.assertEqual(metrics["total_tp"], 1)
        self.assertEqual(metrics["total_fp"], 0)
        self.assertEqual(metrics["total_fn"], 1)
        self.assertIn("total_tn", metrics)
        self.assertIn("pairwise_accuracy", metrics)
        self.assertIn("macro_f05", metrics)

    def test_save_metrics_json_and_csv(self):
        metrics = {
            "macro_f05": 0.9431,
            "macro_precision": 0.9687,
            "macro_recall": 0.8885,
            "total_tp": 6195,
            "total_fp": 126,
            "total_fn": 770,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "metrics.json"
            csv_path = Path(tmp_dir) / "metrics.csv"
            save_metrics(metrics, json_path)
            save_metrics(metrics, csv_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

    def test_baseline_comparison(self):
        current = {"macro_f05": 0.9431, "macro_precision": 0.9687}
        baseline = {"macro_f05": 0.9431, "macro_precision": 0.9600}
        comp = compare_to_baseline(current, baseline)
        self.assertTrue(comp["macro_f05"]["match"])
        self.assertAlmostEqual(comp["macro_precision"]["diff"], 0.0087)


if __name__ == "__main__":
    unittest.main()

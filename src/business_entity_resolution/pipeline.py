"""Central end-to-end Entity Resolution Pipeline for matching S1 reference entities against S2/S3."""
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from .data.loader import DataLoader
from .data.preprocessing import clean_business_name, normalize_text
from .blocking.blocker import EntityBlocker
from .features.feature_engineering import FeatureExtractor
from .models.gbdt import EntityMatcher
from .evaluation.metrics import evaluate_predictions, save_metrics, compare_to_baseline
from .utils import get_project_root


def _to_gt_pairs(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Helper to convert ground truth dataframe into pairwise dataframe."""
    records = []
    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        if matched_str:
            for m in matched_str.split(","):
                m = m.strip()
                if m:
                    records.append({"source1_entity_id": s1, "candidate_entity_id": m})
    return pd.DataFrame(records)


class EntityResolutionPipeline:
    """End-to-end Entity Resolution Pipeline for matching S1 reference entities against S2/S3."""

    def __init__(
        self,
        dataset_dir: str = "dataset",
        output_dir: str = "output",
        model_type: str = "gbdt",
        threshold: float = 0.70
    ):
        p_data = Path(dataset_dir)
        if not p_data.exists() and (Path("..") / dataset_dir).exists():
            p_data = Path("..") / dataset_dir
        self.dataset_dir = p_data

        p_out = Path(output_dir)
        if not p_out.is_absolute() and not Path(".").resolve().name == "entity-reso" and (Path("..") / output_dir).parent.exists():
            p_out = Path("..") / output_dir
        self.output_dir = p_out
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_type = model_type
        self.threshold = threshold
        self.blocker = EntityBlocker(max_block_size=2000)
        self.matcher = EntityMatcher(model_type=model_type, threshold=threshold)

    def build_candidate_pairs(self, s1_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
        """Generate candidate pairs between S1 entities and Reference entities."""
        self.blocker.fit_index(ref_df)
        pairs = []
        for eid, country, name, addr in zip(
            s1_df['entity_id'], s1_df['country'], s1_df['business_name'], s1_df['business_address']
        ):
            cands = self.blocker.get_candidates_for_record(country, name, addr)
            for c in cands:
                pairs.append({"source1_entity_id": eid, "candidate_entity_id": c})
        return pd.DataFrame(pairs)

    def train(
        self,
        train_split: str = "train",
        sample_size: int = 10000,
        random_state: int = 42
    ):
        """Train the entity resolution model on training data."""
        print(f"Loading training data from {self.dataset_dir}/{train_split}...")
        loader = DataLoader(str(self.dataset_dir), train_split)
        gt_full = loader.load_ground_truth()

        sample_gt = gt_full.sample(n=min(sample_size, len(gt_full)), random_state=random_state).reset_index(drop=True)
        sample_s1_ids = set(sample_gt["source1_entity_id"])

        gt_map = {}
        needed_matches = set()
        for _, r in sample_gt.iterrows():
            s1_id = r["source1_entity_id"]
            m = r["matched_entity_ids"]
            matches = set(x.strip() for x in m.split(",") if x.strip()) if m else set()
            gt_map[s1_id] = matches
            needed_matches.update(matches)

        # Load S1 records
        s1 = loader.load_source(1)
        s1_df = s1[s1["entity_id"].isin(sample_s1_ids)].copy().set_index("entity_id")

        # Load needed S2/S3 + background
        s2_rows = []
        s2_path = self.dataset_dir / train_split / f"{train_split}_source2.tsv"
        with open(s2_path, encoding="utf-8") as f:
            header = next(f).strip().split("\t")
            for i, line in enumerate(f):
                p = line.rstrip("\n").split("\t")
                if p[0] in needed_matches or i < 200000:
                    s2_rows.append(p)
        s2_df = pd.DataFrame(s2_rows, columns=header)

        s3_rows = []
        s3_path = self.dataset_dir / train_split / f"{train_split}_source3.tsv"
        with open(s3_path, encoding="utf-8") as f:
            header = next(f).strip().split("\t")
            for i, line in enumerate(f):
                p = line.rstrip("\n").split("\t")
                if p[0] in needed_matches or i < 200000:
                    s3_rows.append(p)
        s3_df = pd.DataFrame(s3_rows, columns=header)

        ref_df = pd.concat([s2_df, s3_df], ignore_index=True)
        print(f"Indexing {len(ref_df):,} reference records for training...")
        self.blocker.fit_index(ref_df)

        # Extract features for training pairs
        X_train, y_train = [], []
        s1_meta = {}
        for eid, country, name, addr in zip(
            s1_df.index, s1_df["country"], s1_df["business_name"], s1_df["business_address"]
        ):
            b_keys, meta = self.blocker.get_blocking_keys_and_meta(country, name, addr)
            s1_meta[eid] = (b_keys, meta)

        print("Building pairwise training features...")
        for s1_id in sample_s1_ids:
            b_keys, s1_data = s1_meta[s1_id]
            s1_country, s1_name, s1_nt, s1_addr, s1_postals, s1_nums, s1_states, s1_words = s1_data
            cands = self.blocker.get_candidates_for_record(
                s1_df.loc[s1_id, "country"],
                s1_df.loc[s1_id, "business_name"],
                s1_df.loc[s1_id, "business_address"]
            )
            true_set = gt_map.get(s1_id, set())

            for ref_id in cands:
                r_data = self.blocker.ref_meta.get(ref_id)
                if not r_data:
                    continue
                r_country, r_name, r_nt, r_addr, r_postals, r_nums, r_states, r_words = r_data

                name_overlap = bool(s1_nt & r_nt)
                addr_word_overlap = bool(s1_words & r_words)
                num_overlap = bool(s1_nums & r_nums)
                postal_overlap = bool(s1_postals & r_postals)

                if s1_nums and r_nums and not num_overlap:
                    if not (s1_nt and r_nt and len(s1_nt & r_nt) >= 2):
                        if not (s1_name and r_name and s1_name[:5] == r_name[:5]):
                            continue

                if not (name_overlap or addr_word_overlap or postal_overlap or (num_overlap and (not r_name or not s1_name))):
                    if not (s1_name and r_name and (s1_name[:4] == r_name[:4])):
                        continue

                feats = FeatureExtractor.extract_features(s1_data, r_data)
                label = 1 if ref_id in true_set else 0
                X_train.append(feats)
                y_train.append(label)

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        print(f"Training LightGBM on {len(X_train):,} pairwise candidate records (Positives: {y_train.sum():,})...")
        self.matcher.fit(X_train, y_train)
        print("Training complete.")

    def predict(
        self,
        split_or_s1: Union[str, pd.DataFrame] = "test",
        ref_df: Optional[pd.DataFrame] = None,
        model_path: Optional[str] = "experiments/results/model.joblib",
        max_records: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate matches and candidate pairs for a set of S1 entities.

        Returns:
            matching_results_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
            candidate_pairs_df: DataFrame with ['source1_entity_id', 'candidate_entity_ids']
        """
        # Ensure model is loaded
        if not hasattr(self.matcher.clf, "fitted_"):
            p_model = Path(model_path) if model_path else None
            if p_model and not p_model.exists() and (Path("..") / model_path).exists():
                p_model = Path("..") / model_path
            if p_model and p_model.exists():
                self.matcher = EntityMatcher.load(p_model)

        if isinstance(split_or_s1, str):
            split = split_or_s1
            loader = DataLoader(str(self.dataset_dir), split)
            s1_df = loader.load_source(1, nrows=max_records)
            if ref_df is None:
                s2 = loader.load_source(2)
                s3 = loader.load_source(3)
                ref_df = pd.concat([s2, s3], ignore_index=True)
        else:
            s1_df = split_or_s1
            if max_records:
                s1_df = s1_df.head(max_records)

        # Index reference records
        blocker = EntityBlocker(max_block_size=2000)
        blocker.fit_index(ref_df)

        matching_rows = []
        candidate_rows = []

        for eid, country, name, addr in zip(
            s1_df["entity_id"], s1_df["country"], s1_df["business_name"], s1_df["business_address"]
        ):
            cands = blocker.get_candidates_for_record(country, name, addr)
            b_keys, s1_data = blocker.get_blocking_keys_and_meta(country, name, addr)
            s1_country, s1_name, s1_nt, s1_addr, s1_postals, s1_nums, s1_states, s1_words = s1_data

            filtered_cands = []
            feats_list = []

            for ref_id in cands:
                r_data = blocker.ref_meta.get(ref_id)
                if not r_data:
                    continue
                r_country, r_name, r_nt, r_addr, r_postals, r_nums, r_states, r_words = r_data

                name_overlap = bool(s1_nt & r_nt)
                addr_word_overlap = bool(s1_words & r_words)
                num_overlap = bool(s1_nums & r_nums)
                postal_overlap = bool(s1_postals & r_postals)

                if s1_nums and r_nums and not num_overlap:
                    if not (s1_nt and r_nt and len(s1_nt & r_nt) >= 2):
                        if not (s1_name and r_name and s1_name[:5] == r_name[:5]):
                            continue

                if not (name_overlap or addr_word_overlap or postal_overlap or (num_overlap and (not r_name or not s1_name))):
                    if not (s1_name and r_name and (s1_name[:4] == r_name[:4])):
                        continue

                filtered_cands.append(ref_id)
                feats = FeatureExtractor.extract_features(s1_data, r_data)
                feats_list.append(feats)

            matched_ids = []
            if feats_list:
                X = np.array(feats_list, dtype=np.float32)
                probs = self.matcher.predict_proba(X)
                for ref_id, prob in zip(filtered_cands, probs):
                    if prob >= self.threshold:
                        matched_ids.append(ref_id)

            matching_rows.append({
                "source1_entity_id": eid,
                "matched_entity_ids": ",".join(matched_ids)
            })
            candidate_rows.append({
                "source1_entity_id": eid,
                "candidate_entity_ids": ",".join(filtered_cands)
            })

        return pd.DataFrame(matching_rows), pd.DataFrame(candidate_rows)

    def evaluate(
        self,
        val_sample_size: int = 10000,
        val_size: int = 2000,
        random_state: int = 42,
        model_path: Optional[str] = "experiments/results/model.joblib",
        baseline_path: Optional[str] = "experiments/results/metrics.json",
        save_metrics_path: Optional[str] = "output/evaluation_metrics.json"
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a strictly held-out validation set.

        Parameters:
            val_sample_size: Total sample size before splitting (e.g. 10,000)
            val_size: Number of held-out validation S1 entities (e.g. 2,000)
            random_state: Seed for reproducible validation split (default 42)
            model_path: Path to trained model joblib file
            baseline_path: Path to baseline metrics for comparison
            save_metrics_path: Path to save evaluation metrics (.json or .csv)

        Returns:
            Dictionary of evaluated metrics and baseline comparison.
        """
        print(f"\n{'='*70}")
        print(f"BUSINESS ENTITY RESOLUTION — MODEL EVALUATION")
        print(f"{'='*70}")
        print(f"Validation split: {val_size:,} entities held out from {val_sample_size:,} sample (seed={random_state})")

        loader = DataLoader(str(self.dataset_dir), "train")
        train_gt, val_gt = loader.get_train_val_split(
            total_sample=val_sample_size,
            val_size=val_size,
            random_state=random_state
        )
        val_s1_ids = set(val_gt["source1_entity_id"])

        # Needed ground truth matches for indexing
        needed_matches = set()
        for _, r in val_gt.iterrows():
            m = r["matched_entity_ids"]
            if m:
                needed_matches.update(x.strip() for x in m.split(",") if x.strip())

        # Load S1 validation entities
        s1_full = loader.load_source(1)
        val_s1_df = s1_full[s1_full["entity_id"].isin(val_s1_ids)].copy().reset_index(drop=True)

        # Load S2 and S3 (all needed matches + 200k background for realistic retrieval)
        print("Loading and indexing reference pool (S2 + S3)...")
        s2_rows = []
        s2_path = self.dataset_dir / "train" / "train_source2.tsv"
        with open(s2_path, encoding="utf-8") as f:
            header = next(f).strip().split("\t")
            for i, line in enumerate(f):
                p = line.rstrip("\n").split("\t")
                if p[0] in needed_matches or i < 200000:
                    s2_rows.append(p)
        s2_df = pd.DataFrame(s2_rows, columns=header)

        s3_rows = []
        s3_path = self.dataset_dir / "train" / "train_source3.tsv"
        with open(s3_path, encoding="utf-8") as f:
            header = next(f).strip().split("\t")
            for i, line in enumerate(f):
                p = line.rstrip("\n").split("\t")
                if p[0] in needed_matches or i < 200000:
                    s3_rows.append(p)
        s3_df = pd.DataFrame(s3_rows, columns=header)

        ref_df = pd.concat([s2_df, s3_df], ignore_index=True)
        print(f"Indexed {len(ref_df):,} reference records across S2 & S3.")

        # Run prediction
        print(f"Running candidate generation & model inference on {len(val_s1_df):,} validation entities...")
        t_start = time.time()
        pred_df, cand_df = self.predict(
            split_or_s1=val_s1_df,
            ref_df=ref_df,
            model_path=model_path
        )
        elapsed = time.time() - t_start
        print(f"Inference completed in {elapsed:.2f}s ({len(val_s1_df)/elapsed:.1f} entities/sec).")

        # Evaluate against validation ground truth
        metrics = evaluate_predictions(
            ground_truth_df=val_gt,
            predictions_df=pred_df,
            candidates_df=cand_df,
            save_path=save_metrics_path,
            baseline_path=baseline_path
        )

        # Also save CSV version if save_metrics_path was json
        if save_metrics_path and save_metrics_path.endswith(".json"):
            csv_path = save_metrics_path.replace(".json", ".csv")
            save_metrics(metrics, csv_path)

        print("\n" + "="*70)
        print("EVALUATION RESULTS SUMMARY:")
        print("="*70)
        print(f"  Macro F0.5 (Primary Metric): {metrics['macro_f05']:.4f}")
        print(f"  Macro Precision:            {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:               {metrics['macro_recall']:.4f}")
        print(f"  Macro F1:                   {metrics['macro_f1']:.4f}")
        print(f"  Exact Accuracy:             {metrics['exact_accuracy']:.4f}")
        if "pairwise_accuracy" in metrics:
            print(f"  Pairwise Accuracy:          {metrics['pairwise_accuracy']:.6f}")
        print(f"  Total Predictions:          {metrics['total_predictions']:,}")
        print(f"  True Positives (TP):        {metrics['total_tp']:,}")
        print(f"  False Positives (FP):       {metrics['total_fp']:,}")
        print(f"  False Negatives (FN):       {metrics['total_fn']:,}")
        if "total_tn" in metrics:
            print(f"  True Negatives (TN):        {metrics['total_tn']:,}")

        if "baseline_comparison" in metrics:
            print("\n" + "-"*70)
            print("COMPARISON WITH PHASE 2 BASELINE:")
            print("-"*70)
            for m_name, comp in metrics["baseline_comparison"].items():
                match_str = "MATCH" if comp["match"] else f"DIFF ({comp['diff']:+.4f})"
                print(f"  {m_name:<20}: Current={comp['current']:.4f} | Baseline={comp['baseline']:.4f} | {match_str}")

        if save_metrics_path:
            print(f"\nSaved metrics to {save_metrics_path} (and .csv)")
        print("="*70 + "\n")

        return metrics

    def run(
        self,
        train_split: str = "train",
        test_split: str = "test",
        save_model_path: Optional[str] = "experiments/results/model.joblib",
        max_test_records: Optional[int] = None
    ):
        """Execute end-to-end training and test inference, outputting validated files."""
        # 1. Train model if not already fitted or load from save_model_path
        if not hasattr(self.matcher.clf, "fitted_"):
            p_model = Path(save_model_path) if save_model_path else None
            if p_model and not p_model.exists() and (Path("..") / save_model_path).exists():
                p_model = Path("..") / save_model_path

            if p_model and p_model.exists():
                print(f"Loading existing trained model from {p_model}...")
                self.matcher = EntityMatcher.load(p_model)
            else:
                self.train(train_split=train_split)
                if save_model_path:
                    self.matcher.save(save_model_path)
                    print(f"Saved model to {save_model_path}")

        # 2. Determine country partitions from test_source1.tsv
        test_s1_path = self.dataset_dir / test_split / f"{test_split}_source1.tsv"
        test_s2_path = self.dataset_dir / test_split / f"{test_split}_source2.tsv"
        test_s3_path = self.dataset_dir / test_split / f"{test_split}_source3.tsv"

        print("Scanning test S1 country list...")
        all_s1_order = []
        country_s1_records: Dict[str, List[Tuple[str, str, str]]] = {}
        with open(test_s1_path, encoding="utf-8") as f:
            next(f)
            count = 0
            for line in f:
                count += 1
                if max_test_records and count > max_test_records:
                    break
                p = line.rstrip("\n").split("\t")
                eid = p[0]
                name = p[1] if len(p) > 1 else ""
                addr = p[2] if len(p) > 2 else ""
                country = p[3] if len(p) > 3 else ""
                all_s1_order.append(eid)
                if country not in country_s1_records:
                    country_s1_records[country] = []
                country_s1_records[country].append((eid, name, addr))

        print(f"Found {len(all_s1_order):,} test S1 entities across {list(country_s1_records.keys())}.")

        all_matches: Dict[str, List[str]] = {}
        all_candidates: Dict[str, List[str]] = {}

        # 3. Process each country partition using a single streaming pass over reference sources
        country_blockers = {c: EntityBlocker(max_block_size=2000) for c in country_s1_records.keys()}
        print(f"Indexing reference records across all {len(country_blockers)} country partitions in a single streaming pass...")
        t_index = time.time()
        for src_path in [test_s2_path, test_s3_path]:
            for chunk in pd.read_csv(src_path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
                for eid, c_try, name, addr in zip(
                    chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]
                ):
                    b = country_blockers.get(c_try)
                    if b is not None:
                        b.index_record(eid, c_try, name, addr)
        print(f"Finished indexing all partitions in {time.time()-t_index:.1f}s.")

        for country, s1_records in country_s1_records.items():
            print(f"\n--- Predicting partition: {country} ({len(s1_records):,} entities) ---")
            country_blocker = country_blockers[country]

            # Predict for S1 entities in this country
            t_pred = time.time()
            batch_count = 0
            for eid, name, addr in s1_records:
                batch_count += 1
                cands = country_blocker.get_candidates_for_record(country, name, addr)
                b_keys, s1_data = country_blocker.get_blocking_keys_and_meta(country, name, addr)
                s1_country, s1_name, s1_nt, s1_addr, s1_postals, s1_nums, s1_states, s1_words = s1_data

                filtered_cands = []
                feats_list = []

                for ref_id in cands:
                    r_data = country_blocker.ref_meta.get(ref_id)
                    if not r_data:
                        continue
                    r_country, r_name, r_nt, r_addr, r_postals, r_nums, r_states, r_words = r_data

                    name_overlap = bool(s1_nt & r_nt)
                    addr_word_overlap = bool(s1_words & r_words)
                    num_overlap = bool(s1_nums & r_nums)
                    postal_overlap = bool(s1_postals & r_postals)

                    if s1_nums and r_nums and not num_overlap:
                        if not (s1_nt and r_nt and len(s1_nt & r_nt) >= 2):
                            if not (s1_name and r_name and s1_name[:5] == r_name[:5]):
                                continue

                    if not (name_overlap or addr_word_overlap or postal_overlap or (num_overlap and (not r_name or not s1_name))):
                        if not (s1_name and r_name and (s1_name[:4] == r_name[:4])):
                            continue

                    filtered_cands.append(ref_id)
                    feats = FeatureExtractor.extract_features(s1_data, r_data)
                    feats_list.append(feats)

                matched_ids = []
                if feats_list:
                    X = np.array(feats_list, dtype=np.float32)
                    probs = self.matcher.predict_proba(X)
                    for ref_id, prob in zip(filtered_cands, probs):
                        if prob >= self.threshold:
                            matched_ids.append(ref_id)

                all_matches[eid] = matched_ids
                all_candidates[eid] = filtered_cands

                if batch_count % 100000 == 0:
                    print(f"  Processed {batch_count:,} / {len(s1_records):,} entities in {time.time()-t_pred:.1f}s...")

            del country_blocker
            gc.collect()

        # 4. Write final output files in exact S1 order
        matching_out_path = self.output_dir / "matching_results.tsv"
        candidate_out_path = self.output_dir / "candidate_pairs.tsv"

        print(f"\nWriting final predictions to {matching_out_path} and {candidate_out_path}...")
        with open(matching_out_path, "w", encoding="utf-8") as f_match, \
             open(candidate_out_path, "w", encoding="utf-8") as f_cand:

            f_match.write("source1_entity_id\tmatched_entity_ids\n")
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

            for s1_id in all_s1_order:
                mids = all_matches.get(s1_id, [])
                cids = all_candidates.get(s1_id, [])
                f_match.write(f"{s1_id}\t{','.join(mids)}\n")
                f_cand.write(f"{s1_id}\t{','.join(cids)}\n")

        print("Finished writing all test predictions.")

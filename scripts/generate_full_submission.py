import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from business_entity_resolution.blocking.blocker import EntityBlocker
from business_entity_resolution.features.extractor import FeatureExtractor
from business_entity_resolution.models.matcher import EntityMatcher

def main():
    print("=" * 80)
    print("STARTING FULL TEST SET INFERENCE (1,732,544 S1 Entities)")
    print("=" * 80)
    t0 = time.time()

    dataset_dir = Path("dataset/test")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = Path("final_submission")
    final_dir.mkdir(parents=True, exist_ok=True)

    # Load trained model
    model_path = Path("experiments/results/model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    print(f"Loading trained model from {model_path}...")
    matcher = EntityMatcher.load(model_path)
    threshold = 0.70
    print(f"Model loaded. Decision threshold: {threshold}")

    test_s1_path = dataset_dir / "test_source1.tsv"
    test_s2_path = dataset_dir / "test_source2.tsv"
    test_s3_path = dataset_dir / "test_source3.tsv"

    # Step 1: Scan S1 entities in original order and group by country
    print("Scanning test_source1.tsv...")
    all_s1_order = []
    country_s1_records: Dict[str, List[Tuple[str, str, str]]] = {}
    with open(test_s1_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            eid = p[0]
            name = p[1] if len(p) > 1 else ""
            addr = p[2] if len(p) > 2 else ""
            country = p[3] if len(p) > 3 else ""
            all_s1_order.append(eid)
            if country not in country_s1_records:
                country_s1_records[country] = []
            country_s1_records[country].append((eid, name, addr))

    print(f"Total S1 entities: {len(all_s1_order):,} across countries: {list(country_s1_records.keys())}")

    all_matches: Dict[str, List[str]] = {}
    all_candidates: Dict[str, List[str]] = {}

    # Step 2: Process each country partition
    for country, s1_records in country_s1_records.items():
        print(f"\n" + "=" * 60)
        print(f"Processing partition: {country} ({len(s1_records):,} S1 entities)")
        print("=" * 60)
        t_country = time.time()

        country_blocker = EntityBlocker(max_block_size=2000)
        indexed_count = 0
        for src_path in [test_s2_path, test_s3_path]:
            t_src = time.time()
            for chunk in pd.read_csv(src_path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
                c_mask = chunk["country"] == country
                c_chunk = chunk[c_mask]
                if len(c_chunk) > 0:
                    for eid, c_try, name, addr in zip(
                        c_chunk["entity_id"], c_chunk["country"], c_chunk["business_name"], c_chunk["business_address"]
                    ):
                        country_blocker.index_record(eid, country, name, addr)
                        indexed_count += 1
            print(f"  Indexed from {src_path.name} in {time.time()-t_src:.1f}s")

        print(f"Total reference records for {country}: {indexed_count:,} (Indexed in {time.time()-t_country:.1f}s)")

        # Batch scoring of S1 records
        BATCH_SIZE = 5000
        t_batch_start = time.time()
        for b_start in range(0, len(s1_records), BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, len(s1_records))
            batch = s1_records[b_start:b_end]

            batch_feats = []
            batch_pairs = [] # (s1_id, ref_id)
            
            for eid, name, addr in batch:
                cands = country_blocker.get_candidates_for_record(country, name, addr)
                b_keys, s1_data = country_blocker.get_blocking_keys_and_meta(country, name, addr)
                s1_country, s1_name, s1_nt, s1_addr, s1_postals, s1_nums, s1_states, s1_words = s1_data

                filtered_cands = []
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
                    batch_feats.append(feats)
                    batch_pairs.append((eid, ref_id))

                all_candidates[eid] = filtered_cands
                all_matches[eid] = []

            # Batch model prediction
            if batch_feats:
                X_batch = np.array(batch_feats, dtype=np.float32)
                probs = matcher.predict_proba(X_batch)
                for (s1_id, ref_id), prob in zip(batch_pairs, probs):
                    if prob >= threshold:
                        all_matches[s1_id].append(ref_id)

            if b_end % 100000 == 0 or b_end == len(s1_records):
                elapsed = time.time() - t_batch_start
                rate = b_end / elapsed if elapsed > 0 else 0
                print(f"  Processed {b_end:,} / {len(s1_records):,} entities ({rate:.0f} entities/sec)...")

        del country_blocker
        gc.collect()

    # Step 3: Write outputs to output/ and final_submission/
    matching_out = output_dir / "matching_results.tsv"
    candidate_out = output_dir / "candidate_pairs.tsv"
    sub_csv = final_dir / "submission.csv"

    print("\n" + "=" * 60)
    print("Writing output files in exact order of test_source1.tsv...")
    print(f"  -> {matching_out}")
    print(f"  -> {candidate_out}")
    print(f"  -> {sub_csv}")
    print("=" * 60)

    with open(matching_out, "w", encoding="utf-8") as f_match, \
         open(candidate_out, "w", encoding="utf-8") as f_cand, \
         open(sub_csv, "w", encoding="utf-8") as f_csv:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_csv.write("source1_entity_id\tmatched_entity_ids\n")

        for s1_id in all_s1_order:
            mids = all_matches.get(s1_id, [])
            cids = all_candidates.get(s1_id, [])
            m_str = ",".join(mids)
            c_str = ",".join(cids)
            f_match.write(f"{s1_id}\t{m_str}\n")
            f_cand.write(f"{s1_id}\t{c_str}\n")
            f_csv.write(f"{s1_id}\t{m_str}\n")

    print(f"ALL FILES WRITTEN SUCCESSFULLY in {time.time()-t0:.2f}s!")

if __name__ == "__main__":
    main()

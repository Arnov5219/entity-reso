import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import re
import time
import json
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from business_entity_resolution.data.preprocessing import clean_business_name, normalize_text, US_STATES, IN_STATES
from business_entity_resolution.evaluation.metrics import evaluate_predictions
import lightgbm as lgb

print("="*80)
print("EXPERIMENT 4: Scaling to 10,000 entities (8,000 Train, 2,000 Validation)")
print("="*80)
t0 = time.time()

gt_full = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', keep_default_na=False)

# Strictly split entities by source1_entity_id: 8,000 train, 2,000 validation
sample = gt_full.sample(n=10000, random_state=42).reset_index(drop=True)
train_gt = sample.iloc[:8000].reset_index(drop=True)
val_gt = sample.iloc[8000:].reset_index(drop=True)

train_s1_ids = set(train_gt['source1_entity_id'])
val_s1_ids = set(val_gt['source1_entity_id'])
all_s1_ids = train_s1_ids | val_s1_ids

gt_map = {}
needed_matches = set()
for _, r in sample.iterrows():
    s1_id = r['source1_entity_id']
    m = r['matched_entity_ids']
    matches = set(x.strip() for x in m.split(',') if x.strip()) if m else set()
    gt_map[s1_id] = matches
    needed_matches.update(matches)

# Load S1
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', keep_default_na=False)
s1_df = s1[s1['entity_id'].isin(all_s1_ids)].copy().set_index('entity_id')

# Load S2 and S3: all needed matches + 200k random background records each
s2_rows = []
with open('dataset/train/train_source2.tsv', encoding='utf-8') as f:
    header = next(f).strip().split('\t')
    for i, line in enumerate(f):
        p = line.rstrip('\n').split('\t')
        if p[0] in needed_matches or i < 200000:
            s2_rows.append(p)
s2 = pd.DataFrame(s2_rows, columns=header)

s3_rows = []
with open('dataset/train/train_source3.tsv', encoding='utf-8') as f:
    header = next(f).strip().split('\t')
    for i, line in enumerate(f):
        p = line.rstrip('\n').split('\t')
        if p[0] in needed_matches or i < 200000:
            s3_rows.append(p)
s3 = pd.DataFrame(s3_rows, columns=header)

ref = pd.concat([s2, s3], ignore_index=True).set_index('entity_id')
print(f"Loaded {len(ref)} reference records in {time.time()-t0:.2f}s.")

ALL_STATES = set(US_STATES.values()) | set(IN_STATES.values())
STOPWORDS = {"the", "and", "of", "in", "for", "a", "an", "on", "at", "to", "by", "with", "from", "service", "services", "center", "enterprises", "partners"}
COMMON_ADDR_WORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr", "court", "ct",
    "floor", "fl", "building", "bldg", "block", "blk", "sector", "sec", "phase", "plot",
    "nagar", "colony", "apartment", "apartments", "apt", "near", "opp", "opposite", "behind",
    "main", "cross", "north", "south", "east", "west", "india", "delhi", "bangalore", "mumbai",
    "plaza", "tower", "complex", "house", "circle", "square", "park", "estate", "state", "city",
    "unit", "suite", "room", "first", "second", "third", "fourth", "fifth", "village", "post"
} | ALL_STATES

def clean_name_tokens(text):
    text = re.sub(r'[\*\#\>\<\[\]]', ' ', str(text))
    clean = clean_business_name(text)
    clean = re.sub(r'\.(com|org|net|in|co|io)\b', '', clean)
    tokens = [t for t in clean.split() if len(t) > 1 and t not in STOPWORDS]
    return clean, tokens

def get_address_info(text):
    if not text:
        return "", set(), set(), set(), set()
    norm = normalize_text(text)
    norm = re.sub(r'\b(h\s*no|hn|c\s*o|plot\s*no|flat\s*no|door\s*no|unit\s*unit|unit)\b', ' ', norm)
    postals = set(re.findall(r'\b\d{5,6}\b', norm))
    nums = set(re.findall(r'\d+', norm))
    tokens = norm.split()
    states = {US_STATES.get(t, IN_STATES.get(t, t)) for t in tokens if t in US_STATES or t in IN_STATES or t in ALL_STATES}
    words = {t for t in tokens if len(t) >= 4 and t not in COMMON_ADDR_WORDS and t not in STOPWORDS}
    return norm, postals, nums, states, words

def get_keys_and_meta(country, name, addr):
    country = str(country).strip()
    clean_name, nt = clean_name_tokens(name)
    norm_addr, postals, nums, states, words = get_address_info(addr)
    keys = set()
    
    if nt:
        w0 = nt[0]
        if len(w0) >= 3:
            keys.add(f"N:{country}:{w0}")
            if len(w0) >= 6:
                keys.add(f"NP:{country}:{w0[:5]}")
        if len(nt) >= 2:
            w1 = nt[1]
            if len(w1) >= 4:
                keys.add(f"N:{country}:{w1}")
            keys.add(f"N2:{country}:{w0[:4]}_{w1[:4]}")
            
    if nums and words:
        num = next(iter(nums))
        for w in list(words)[:2]:
            keys.add(f"AN:{country}:{num}_{w}")
            
    for w in list(words)[:3]:
        if len(w) >= 6:
            keys.add(f"AW:{country}:{w}")
            
    for p in postals:
        keys.add(f"P:{country}:{p}")
        
    return keys, (country, clean_name, set(nt), norm_addr, postals, nums, states, words)

print("Indexing reference pool...")
t1 = time.time()
block_index = {}
ref_meta = {}
for eid, country, name, addr in zip(ref.index, ref['country'], ref['business_name'], ref['business_address']):
    b_keys, meta = get_keys_and_meta(country, name, addr)
    ref_meta[eid] = meta
    for k in b_keys:
        if k not in block_index:
            block_index[k] = []
        block_index[k].append(eid)
print(f"Indexed in {time.time()-t1:.2f}s with {len(block_index)} keys.")

s1_meta = {}
for eid, country, name, addr in zip(s1_df.index, s1_df['country'], s1_df['business_name'], s1_df['business_address']):
    b_keys, meta = get_keys_and_meta(country, name, addr)
    s1_meta[eid] = (b_keys, meta)

def extract_features(s1_data, r_data):
    s1_country, s1_name, s1_nt, s1_addr, s1_postals, s1_nums, s1_states, s1_words = s1_data
    r_country, r_name, r_nt, r_addr, r_postals, r_nums, r_states, r_words = r_data
    
    n_ratio = fuzz.ratio(s1_name, r_name) / 100.0 if (s1_name and r_name) else 0.0
    n_sort = fuzz.token_sort_ratio(s1_name, r_name) / 100.0 if (s1_name and r_name) else 0.0
    n_set = fuzz.token_set_ratio(s1_name, r_name) / 100.0 if (s1_name and r_name) else 0.0
    n_jacc = len(s1_nt & r_nt) / len(s1_nt | r_nt) if (s1_nt or r_nt) else 0.0
    n_contain = len(s1_nt & r_nt) / min(len(s1_nt), len(r_nt)) if (s1_nt and r_nt) else 0.0
    n_exact = 1.0 if (s1_name and r_name and s1_name == r_name) else 0.0
    
    s1_nospace = s1_name.replace(" ", "")
    r_nospace = r_name.replace(" ", "")
    dom_m = 1.0 if (s1_nospace and r_nospace and (s1_nospace in r_nospace or r_nospace in s1_nospace)) else 0.0
    
    a_empty = 1.0 if not r_addr else 0.0
    a_ratio = fuzz.ratio(s1_addr, r_addr) / 100.0 if (s1_addr and r_addr) else 0.0
    a_sort = fuzz.token_sort_ratio(s1_addr, r_addr) / 100.0 if (s1_addr and r_addr) else 0.0
    a_set = fuzz.token_set_ratio(s1_addr, r_addr) / 100.0 if (s1_addr and r_addr) else 0.0
    a_jacc = len(s1_words & r_words) / len(s1_words | r_words) if (s1_words or r_words) else 0.0
    a_contain = len(s1_words & r_words) / min(len(s1_words), len(r_words)) if (s1_words and r_words) else 0.0
    
    if s1_nums and r_nums:
        shared_nums = s1_nums & r_nums
        num_match = 1.0 if bool(shared_nums) else 0.0
        num_mismatch = 1.0 if not bool(shared_nums) else 0.0
    else:
        num_match = 0.5
        num_mismatch = 0.0
        
    if s1_postals and r_postals:
        postal_match = 1.0 if bool(s1_postals & r_postals) else 0.0
        postal_mismatch = 1.0 if not bool(s1_postals & r_postals) else 0.0
    else:
        postal_match = 0.5
        postal_mismatch = 0.0
        
    if s1_states and r_states:
        state_match = 1.0 if bool(s1_states & r_states) else 0.0
        state_mismatch = 1.0 if not bool(s1_states & r_states) else 0.0
    else:
        state_match = 0.5
        state_mismatch = 0.0
        
    name_match_empty_addr = 1.0 if (a_empty == 1.0 and n_sort >= 0.85) else 0.0
    addr_match_diff_name = 1.0 if (a_set >= 0.80 and num_match == 1.0 and n_ratio < 0.50) else 0.0
    both_strong = 1.0 if (n_sort >= 0.75 and a_set >= 0.70 and num_mismatch == 0.0) else 0.0
    
    return [
        n_ratio, n_sort, n_set, n_jacc, n_contain, n_exact, dom_m,
        a_empty, a_ratio, a_sort, a_set, a_jacc, a_contain,
        num_match, num_mismatch, postal_match, postal_mismatch,
        state_match, state_mismatch,
        name_match_empty_addr, addr_match_diff_name, both_strong
    ]

def build_dataset(s1_subset_ids):
    X = []
    y = []
    pairs = []
    for s1_id in s1_subset_ids:
        b_keys, s1_data = s1_meta[s1_id]
        s1_country, s1_name, s1_nt, s1_addr, s1_postals, s1_nums, s1_states, s1_words = s1_data
        
        cands = set()
        for k in b_keys:
            lst = block_index.get(k)
            if lst and len(lst) <= 2000:
                cands.update(lst)
                
        true_set = gt_map.get(s1_id, set())
        
        for ref_id in cands:
            r_data = ref_meta[ref_id]
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
                    
            feats = extract_features(s1_data, r_data)
            label = 1 if ref_id in true_set else 0
            X.append(feats)
            y.append(label)
            pairs.append((s1_id, ref_id))
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), pairs

print("Building train dataset (8,000 S1 entities)...")
t2 = time.time()
X_train, y_train, train_pairs = build_dataset(train_s1_ids)
print(f"Train: {len(X_train)} pairs (Positives: {y_train.sum()}, Negatives: {len(y_train)-y_train.sum()}) in {time.time()-t2:.2f}s")

print("Building val dataset (2,000 S1 entities)...")
t3 = time.time()
X_val, y_val, val_pairs = build_dataset(val_s1_ids)
print(f"Val: {len(X_val)} pairs (Positives: {y_val.sum()}, Negatives: {len(y_val)-y_val.sum()}) in {time.time()-t3:.2f}s")

print("Training production LightGBM...")
t4 = time.time()
clf = lgb.LGBMClassifier(
    n_estimators=350,
    learning_rate=0.06,
    num_leaves=63,
    max_depth=9,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
print(f"Trained in {time.time()-t4:.2f}s.")

val_probs = clf.predict_proba(X_val)[:, 1]

best_thresh = 0.65
best_f05 = 0.0
best_m = {}

print("\n--- Validation Threshold Search ---")
for thresh in np.arange(0.40, 0.85, 0.05):
    preds_by_s1 = {s1_id: [] for s1_id in val_s1_ids}
    for (s1_id, ref_id), prob in zip(val_pairs, val_probs):
        if prob >= thresh:
            preds_by_s1[s1_id].append(ref_id)
            
    pred_df = pd.DataFrame([
        {'source1_entity_id': s1_id, 'matched_entity_ids': ','.join(mids)}
        for s1_id, mids in preds_by_s1.items()
    ])
    
    m = evaluate_predictions(val_gt, pred_df)
    pairwise_acc = ( (val_probs >= thresh) == y_val ).mean()
    print(f"Thresh {thresh:.2f} | Macro F0.5: {m['macro_f05']:.4f} | Prec: {m['macro_precision']:.4f} | Rec: {m['macro_recall']:.4f} | F1: {m['macro_f1']:.4f} | Pairwise Acc: {pairwise_acc:.4f} | TP: {m['total_tp']} | FP: {m['total_fp']} | FN: {m['total_fn']}")
    if m['macro_f05'] > best_f05:
        best_f05 = m['macro_f05']
        best_thresh = thresh
        best_m = m
        best_m['pairwise_accuracy'] = float(pairwise_acc)

print("\n" + "="*80)
print(f"FINAL BEST METRICS at Threshold {best_thresh:.2f}:")
for k, v in best_m.items():
    print(f"  {k}: {v}")

# Save model and metrics to experiments
import os
os.makedirs('experiments/results', exist_ok=True)
joblib.dump(clf, 'experiments/results/model.joblib')
with open('experiments/results/metrics.json', 'w') as f:
    json.dump(best_m, f, indent=2)
print("Saved experiments/results/model.joblib and experiments/results/metrics.json")

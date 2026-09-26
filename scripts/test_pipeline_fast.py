import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import re
import time
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from business_entity_resolution.data.preprocessing import clean_business_name, normalize_text
from business_entity_resolution.evaluation.metrics import evaluate_predictions
import lightgbm as lgb

print("Testing Fast ML Pipeline on 2,500 validation entities...")
t0 = time.time()
gt_full = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', keep_default_na=False)

# 2500 total: 1500 train, 1000 validation
sample = gt_full.sample(n=2500, random_state=42).reset_index(drop=True)
train_gt = sample.iloc[:1500].reset_index(drop=True)
val_gt = sample.iloc[1500:].reset_index(drop=True)

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

# Load S2 and S3: all needed matches + 100k random background records each
s2_rows = []
with open('dataset/train/train_source2.tsv', encoding='utf-8') as f:
    header = next(f).strip().split('\t')
    for i, line in enumerate(f):
        p = line.rstrip('\n').split('\t')
        if p[0] in needed_matches or i < 100000:
            s2_rows.append(p)
s2 = pd.DataFrame(s2_rows, columns=header)

s3_rows = []
with open('dataset/train/train_source3.tsv', encoding='utf-8') as f:
    header = next(f).strip().split('\t')
    for i, line in enumerate(f):
        p = line.rstrip('\n').split('\t')
        if p[0] in needed_matches or i < 100000:
            s3_rows.append(p)
s3 = pd.DataFrame(s3_rows, columns=header)

ref = pd.concat([s2, s3], ignore_index=True).set_index('entity_id')
print(f"Loaded in {time.time()-t0:.2f}s. Reference: {len(ref)} records.")

STOPWORDS = {"the", "and", "of", "in", "for", "a", "an", "on", "at", "to", "by", "with", "from", "service", "services", "center", "enterprises", "partners"}
COMMON_ADDR_WORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr", "court", "ct",
    "floor", "fl", "building", "bldg", "block", "blk", "sector", "sec", "phase", "plot",
    "nagar", "colony", "apartment", "apartments", "apt", "near", "opp", "opposite", "behind",
    "main", "cross", "north", "south", "east", "west", "india", "delhi", "bangalore", "mumbai",
    "plaza", "tower", "complex", "house", "circle", "square", "park", "estate", "state", "city",
    "unit", "suite", "room", "first", "second", "third", "fourth", "fifth", "village", "post"
}

def clean_name_tokens(text):
    text = re.sub(r'[\*\#\>\<\[\]]', ' ', str(text))
    clean = clean_business_name(text)
    clean = re.sub(r'\.(com|org|net|in|co|io)\b', '', clean)
    tokens = [t for t in clean.split() if len(t) > 1 and t not in STOPWORDS]
    return clean, tokens

def get_address_info(text):
    if not text:
        return "", set(), set()
    norm = normalize_text(text)
    norm = re.sub(r'\b(h\s*no|hn|c\s*o|plot\s*no|flat\s*no|door\s*no|unit\s*unit|unit)\b', ' ', norm)
    tokens = norm.split()
    nums = {t for t in tokens if any(c.isdigit() for c in t)}
    words = {t for t in tokens if len(t) >= 4 and t not in COMMON_ADDR_WORDS and t not in STOPWORDS}
    return norm, nums, words

def get_keys_and_meta(country, name, addr):
    country = str(country).strip()
    clean_name, nt = clean_name_tokens(name)
    norm_addr, nums, words = get_address_info(addr)
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
            
    return keys, (country, clean_name, set(nt), norm_addr, nums, words)

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

# Fast candidate generation with light pre-filter
def build_dataset(s1_subset_ids):
    X = []
    y = []
    pairs = []
    
    for s1_id in s1_subset_ids:
        b_keys, s1_data = s1_meta[s1_id]
        s1_country, s1_name, s1_nt, s1_addr, s1_nums, s1_words = s1_data
        
        cands = set()
        for k in b_keys:
            lst = block_index.get(k)
            if lst and len(lst) <= 1500:
                cands.update(lst)
                
        true_set = gt_map.get(s1_id, set())
        
        # Fast filter on candidates:
        # Candidate must have:
        # (a) at least 1 shared name token, OR
        # (b) at least 1 shared address word or shared number, OR
        # (c) address is empty and name prefix matches
        for ref_id in cands:
            r_country, r_name, r_nt, r_addr, r_nums, r_words = ref_meta[ref_id]
            
            # Fast check
            name_overlap = bool(s1_nt & r_nt)
            addr_word_overlap = bool(s1_words & r_words)
            num_overlap = bool(s1_nums & r_nums)
            
            # If neither shares words and names are completely different, skip
            if not (name_overlap or addr_word_overlap or (num_overlap and (not r_name or not s1_name))):
                # Check rapid prefix/substring
                if not (s1_name and r_name and (s1_name[:4] == r_name[:4])):
                    continue
            
            # Compute ML features
            n_ratio = fuzz.ratio(s1_name, r_name) / 100.0
            n_sort = fuzz.token_sort_ratio(s1_name, r_name) / 100.0
            n_set = fuzz.token_set_ratio(s1_name, r_name) / 100.0
            n_jacc = len(s1_nt & r_nt) / len(s1_nt | r_nt) if (s1_nt or r_nt) else 0.0
            
            a_ratio = fuzz.ratio(s1_addr, r_addr) / 100.0 if (s1_addr and r_addr) else 0.0
            a_set = fuzz.token_set_ratio(s1_addr, r_addr) / 100.0 if (s1_addr and r_addr) else 0.0
            a_jacc = len(s1_words & r_words) / len(s1_words | r_words) if (s1_words or r_words) else 0.0
            
            num_m = 1.0 if (s1_nums and r_nums and bool(s1_nums & r_nums)) else (0.5 if (not s1_nums or not r_nums) else 0.0)
            a_empty = 1.0 if not r_addr else 0.0
            
            s1_nospace = s1_name.replace(" ", "")
            r_nospace = r_name.replace(" ", "")
            dom_m = 1.0 if (s1_nospace and r_nospace and (s1_nospace in r_nospace or r_nospace in s1_nospace)) else 0.0
            
            label = 1 if ref_id in true_set else 0
            feats = [n_ratio, n_sort, n_set, n_jacc, a_ratio, a_set, a_jacc, num_m, a_empty, dom_m]
            
            X.append(feats)
            y.append(label)
            pairs.append((s1_id, ref_id))
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), pairs

print("Building train dataset...")
t2 = time.time()
X_train, y_train, train_pairs = build_dataset(train_s1_ids)
print(f"Train: {len(X_train)} pairs (Positives: {y_train.sum()}, Negatives: {len(y_train)-y_train.sum()}) in {time.time()-t2:.2f}s")

print("Building val dataset...")
t3 = time.time()
X_val, y_val, val_pairs = build_dataset(val_s1_ids)
print(f"Val: {len(X_val)} pairs (Positives: {y_val.sum()}, Negatives: {len(y_val)-y_val.sum()}) in {time.time()-t3:.2f}s")

# Train LightGBM
print("Training LightGBM...")
t4 = time.time()
clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
print(f"Trained in {time.time()-t4:.2f}s.")

# Validate
val_probs = clf.predict_proba(X_val)[:, 1]

best_thresh = 0.5
best_f05 = 0.0
best_m = {}

print("\n--- Validation Threshold Tuning ---")
for thresh in np.arange(0.25, 0.85, 0.05):
    preds_by_s1 = {s1_id: [] for s1_id in val_s1_ids}
    for (s1_id, ref_id), prob in zip(val_pairs, val_probs):
        if prob >= thresh:
            preds_by_s1[s1_id].append(ref_id)
            
    pred_df = pd.DataFrame([
        {'source1_entity_id': s1_id, 'matched_entity_ids': ','.join(mids)}
        for s1_id, mids in preds_by_s1.items()
    ])
    
    m = evaluate_predictions(val_gt, pred_df)
    print(f"Thresh {thresh:.2f} | Macro F0.5: {m['macro_f05']:.4f} | Accuracy: {m['exact_accuracy']:.4f} | Prec: {m['macro_precision']:.4f} | Rec: {m['macro_recall']:.4f} | F1: {m['macro_f1']:.4f}")
    if m['macro_f05'] > best_f05:
        best_f05 = m['macro_f05']
        best_thresh = thresh
        best_m = m

print("\n" + "="*80)
print(f"BEST RESULT at Threshold {best_thresh:.2f}:")
for k, v in best_m.items():
    print(f"  {k}: {v}")

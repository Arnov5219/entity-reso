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

print("Running feature engineering & matching experiment...")
t0 = time.time()
gt_full = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', keep_default_na=False)

# Let's split 3000 S1 entities into: 2000 train, 1000 val
s1_sample = gt_full.sample(n=3000, random_state=42).reset_index(drop=True)
train_gt = s1_sample.iloc[:2000].reset_index(drop=True)
val_gt = s1_sample.iloc[2000:].reset_index(drop=True)

train_s1_ids = set(train_gt['source1_entity_id'])
val_s1_ids = set(val_gt['source1_entity_id'])
all_s1_ids = train_s1_ids | val_s1_ids

gt_map = {}
needed_matches = set()
for _, r in s1_sample.iterrows():
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
print(f"Reference pool: {len(ref)} records. Loading took {time.time()-t0:.2f}s")

# Blocking functions
STOPWORDS = {"the", "and", "of", "in", "for", "a", "an", "on", "at", "to", "by", "with", "from", "service", "services", "center", "enterprises", "partners"}
COMMON_ADDR_WORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr", "court", "ct",
    "floor", "fl", "building", "bldg", "block", "blk", "sector", "sec", "phase", "plot",
    "nagar", "colony", "apartment", "apartments", "apt", "near", "opp", "opposite", "behind",
    "main", "cross", "north", "south", "east", "west", "india", "delhi", "bangalore", "mumbai"
}

def clean_name_tokens(text):
    text = re.sub(r'[\*\#\>\<\[\]]', ' ', str(text))
    clean = clean_business_name(text)
    clean = re.sub(r'\.(com|org|net|in|co|io)\b', '', clean)
    tokens = [t for t in clean.split() if len(t) > 1 and t not in STOPWORDS]
    return clean, tokens

def get_address_info(text):
    if not text:
        return "", [], []
    norm = normalize_text(text)
    norm = re.sub(r'\b(h\s*no|hn|c\s*o|plot\s*no|flat\s*no|door\s*no|unit\s*unit|unit)\b', ' ', norm)
    tokens = norm.split()
    nums = [t for t in tokens if any(c.isdigit() for c in t)]
    words = [t for t in tokens if len(t) >= 4 and t not in COMMON_ADDR_WORDS and t not in STOPWORDS]
    return norm, nums, words

def get_blocking_keys(country, name, addr):
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
            
    for w in words[:4]:
        if len(w) >= 5:
            keys.add(f"A:{country}:{w}")
            
    if nums and words:
        keys.add(f"AN:{country}:{nums[0]}_{words[0][:4]}")
        
    return keys, clean_name, nt, norm_addr, nums, words

print("Indexing reference pool...")
t1 = time.time()
block_index = {}
ref_meta = {}

for eid, country, name, addr in zip(ref.index, ref['country'], ref['business_name'], ref['business_address']):
    b_keys, c_name, nt, n_addr, nums, words = get_blocking_keys(country, name, addr)
    ref_meta[eid] = (country, c_name, set(nt), n_addr, set(nums), set(words))
    for k in b_keys:
        if k not in block_index:
            block_index[k] = []
        block_index[k].append(eid)

print(f"Indexed in {time.time()-t1:.2f}s.")

def extract_pairwise_features(s1_data, ref_data):
    # s1_data: (country, c_name, nt_set, n_addr, nums_set, words_set)
    s1_country, s1_name, s1_nt, s1_addr, s1_nums, s1_words = s1_data
    r_country, r_name, r_nt, r_addr, r_nums, r_words = ref_data
    
    # 1. Name ratios
    name_ratio = fuzz.ratio(s1_name, r_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, r_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, r_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, r_name) / 100.0
    
    # Name Jaccard
    name_jaccard = len(s1_nt & r_nt) / len(s1_nt | r_nt) if (s1_nt or r_nt) else 0.0
    
    # 2. Address ratios
    addr_ratio = fuzz.ratio(s1_addr, r_addr) / 100.0 if (s1_addr and r_addr) else 0.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, r_addr) / 100.0 if (s1_addr and r_addr) else 0.0
    addr_jaccard = len(s1_words & r_words) / len(s1_words | r_words) if (s1_words or r_words) else 0.0
    
    # Number overlap
    num_match = 1.0 if (s1_nums and r_nums and bool(s1_nums & r_nums)) else (0.5 if (not s1_nums or not r_nums) else 0.0)
    
    # Missingness
    addr_empty = 1.0 if not r_addr else 0.0
    
    # Domain / concatenated match
    s1_nospace = s1_name.replace(" ", "")
    r_nospace = r_name.replace(" ", "")
    domain_match = 1.0 if (s1_nospace and r_nospace and (s1_nospace in r_nospace or r_nospace in s1_nospace)) else 0.0
    
    return [
        name_ratio,
        name_token_sort,
        name_token_set,
        name_partial,
        name_jaccard,
        addr_ratio,
        addr_token_set,
        addr_jaccard,
        num_match,
        addr_empty,
        domain_match
    ]

# Generate candidate pairs for train and val
def generate_candidates(s1_subset_ids):
    X = []
    y = []
    pair_ids = [] # (s1_id, ref_id)
    
    for s1_id in s1_subset_ids:
        r = s1_df.loc[s1_id]
        b_keys, c_name, nt, n_addr, nums, words = get_blocking_keys(r['country'], r['business_name'], r['business_address'])
        s1_data = (r['country'], c_name, set(nt), n_addr, set(nums), set(words))
        
        cands = set()
        for k in b_keys:
            lst = block_index.get(k)
            if lst and len(lst) < 5000:
                cands.update(lst)
                
        true_matches = gt_map.get(s1_id, set())
        for ref_id in cands:
            ref_data = ref_meta[ref_id]
            feats = extract_pairwise_features(s1_data, ref_data)
            label = 1 if ref_id in true_matches else 0
            X.append(feats)
            y.append(label)
            pair_ids.append((s1_id, ref_id))
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), pair_ids

print("Generating training candidate pairs...")
t2 = time.time()
X_train, y_train, train_pairs = generate_candidates(train_s1_ids)
print(f"Train pairs: {len(X_train)} (Positives: {y_train.sum()}, Negatives: {len(y_train)-y_train.sum()}) in {time.time()-t2:.2f}s")

print("Generating validation candidate pairs...")
t3 = time.time()
X_val, y_val, val_pairs = generate_candidates(val_s1_ids)
print(f"Val pairs: {len(X_val)} (Positives: {y_val.sum()}, Negatives: {len(y_val)-y_val.sum()}) in {time.time()-t3:.2f}s")

# Train LightGBM classifier
import lightgbm as lgb
print("Training LightGBM model...")
t4 = time.time()
model = lgb.LGBMClassifier(
    n_estimators=150,
    learning_rate=0.08,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    importance_type='gain'
)
model.fit(X_train, y_train)
print(f"Model trained in {time.time()-t4:.2f}s.")

# Predict probabilities on val
val_probs = model.predict_proba(X_val)[:, 1]

# Threshold search on validation
best_thresh = 0.5
best_f05 = 0.0
best_metrics = {}

print("Evaluating decision thresholds on validation set...")
for thresh in np.arange(0.30, 0.85, 0.05):
    preds_by_s1 = {s1_id: [] for s1_id in val_s1_ids}
    for (s1_id, ref_id), prob in zip(val_pairs, val_probs):
        if prob >= thresh:
            preds_by_s1[s1_id].append(ref_id)
            
    val_pred_df = pd.DataFrame([
        {'source1_entity_id': s1_id, 'matched_entity_ids': ','.join(mids)}
        for s1_id, mids in preds_by_s1.items()
    ])
    
    m = evaluate_predictions(val_gt, val_pred_df)
    print(f"Thresh {thresh:.2f} -> Macro F0.5: {m['macro_f05']:.4f}, Accuracy: {m['exact_accuracy']:.4f}, Prec: {m['macro_precision']:.4f}, Rec: {m['macro_recall']:.4f}, F1: {m['macro_f1']:.4f}")
    if m['macro_f05'] > best_f05:
        best_f05 = m['macro_f05']
        best_thresh = thresh
        best_metrics = m

print("\n" + "="*80)
print(f"BEST THRESHOLD: {best_thresh:.2f}")
print("BEST METRICS:")
for k, v in best_metrics.items():
    print(f"  {k}: {v}")

# Feature importances
feature_names = [
    "name_ratio", "name_token_sort", "name_token_set", "name_partial",
    "name_jaccard", "addr_ratio", "addr_token_set", "addr_jaccard",
    "num_match", "addr_empty", "domain_match"
]
imp = model.feature_importances_
print("\nFeature Importances:")
for name, score in sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {score:.2f}")

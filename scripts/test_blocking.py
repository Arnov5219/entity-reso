import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import re
import time
import pandas as pd
from business_entity_resolution.data.preprocessing import clean_business_name, normalize_text

print("Loading test data for blocking experiment...")
t0 = time.time()
gt_full = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', keep_default_na=False)
val_sample = gt_full.sample(n=2000, random_state=42).reset_index(drop=True)
val_s1_ids = set(val_sample['source1_entity_id'])

# Map S1 to true match IDs
gt_map = {}
total_true_matches = 0
needed_s2 = set()
needed_s3 = set()
for _, r in val_sample.iterrows():
    s1_id = r['source1_entity_id']
    m = r['matched_entity_ids']
    matches = set(x.strip() for x in m.split(',') if x.strip()) if m else set()
    gt_map[s1_id] = matches
    total_true_matches += len(matches)
    for x in matches:
        if x.startswith('S2-'): needed_s2.add(x)
        elif x.startswith('S3-'): needed_s3.add(x)

print(f"Sample: {len(val_sample)} S1 entities with {total_true_matches} true matches.")

# Load S1
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', keep_default_na=False)
val_s1 = s1[s1['entity_id'].isin(val_s1_ids)].copy()

# Load S2 and S3: all needed matches + 100k random background records each
s2_rows = []
with open('dataset/train/train_source2.tsv', encoding='utf-8') as f:
    header = next(f).strip().split('\t')
    for i, line in enumerate(f):
        p = line.rstrip('\n').split('\t')
        if p[0] in needed_s2 or i < 100000:
            s2_rows.append(p)
s2 = pd.DataFrame(s2_rows, columns=header)

s3_rows = []
with open('dataset/train/train_source3.tsv', encoding='utf-8') as f:
    header = next(f).strip().split('\t')
    for i, line in enumerate(f):
        p = line.rstrip('\n').split('\t')
        if p[0] in needed_s3 or i < 100000:
            s3_rows.append(p)
s3 = pd.DataFrame(s3_rows, columns=header)

ref = pd.concat([s2, s3], ignore_index=True)
print(f"Reference pool: {len(ref)} records ({len(s2)} S2, {len(s3)} S3). Loading took {time.time()-t0:.2f}s")

# Let's test blocking functions
STOPWORDS = {"the", "and", "of", "in", "for", "a", "an", "on", "at", "to", "by", "with", "from"}

def get_name_tokens(text):
    clean = clean_business_name(text)
    # Also handle domains like foo.com -> foo
    clean = re.sub(r'\.(com|org|net|in|co|io)\b', '', clean)
    tokens = [t for t in clean.split() if len(t) > 1 and t not in STOPWORDS]
    return tokens

def get_address_keys(text):
    if not text:
        return []
    norm = normalize_text(text)
    tokens = norm.split()
    # Find numbers
    nums = [t for t in tokens if any(c.isdigit() for c in t)]
    words = [t for t in tokens if t.isalpha() and len(t) > 2 and t not in STOPWORDS]
    keys = []
    # If there is a number and a word: (num, word)
    if nums and words:
        num = nums[0]
        for w in words[:2]:
            keys.append(f"{num}_{w}")
    return keys

def get_blocking_keys(row):
    country = str(row['country']).strip()
    keys = set()
    
    # 1. First 2 significant name tokens
    name_tokens = get_name_tokens(row['business_name'])
    if name_tokens:
        # First token if length >= 3
        if len(name_tokens[0]) >= 3:
            keys.add(f"N1:{country}:{name_tokens[0]}")
        if len(name_tokens) >= 2:
            keys.add(f"N2:{country}:{name_tokens[0]}_{name_tokens[1]}")
            if len(name_tokens[1]) >= 4:
                keys.add(f"N1:{country}:{name_tokens[1]}")
                
    # 2. Address keys: number + street token
    addr_keys = get_address_keys(row['business_address'])
    for ak in addr_keys:
        keys.add(f"A:{country}:{ak}")
        
    return keys

print("Building index for reference pool...")
t1 = time.time()
block_index = {}
for idx, r in ref.iterrows():
    b_keys = get_blocking_keys(r)
    eid = r['entity_id']
    for k in b_keys:
        if k not in block_index:
            block_index[k] = []
        block_index[k].append(eid)
print(f"Index built in {time.time()-t1:.2f}s with {len(block_index)} distinct keys.")

# Query S1 against index
print("Querying S1 candidates...")
t2 = time.time()
total_candidates = 0
found_matches = 0
candidate_dict = {}

for idx, r in val_s1.iterrows():
    s1_id = r['entity_id']
    b_keys = get_blocking_keys(r)
    cands = set()
    for k in b_keys:
        if k in block_index:
            cands.update(block_index[k])
    
    candidate_dict[s1_id] = cands
    total_candidates += len(cands)
    
    true_set = gt_map.get(s1_id, set())
    found_matches += len(true_set & cands)

print(f"Querying took {time.time()-t2:.2f}s")
recall = found_matches / total_true_matches if total_true_matches > 0 else 0
avg_candidates = total_candidates / len(val_s1)
print(f"=== BLOCKING RESULTS ===")
print(f"Total True Matches: {total_true_matches}")
print(f"Found Matches in Candidates: {found_matches}")
print(f"Candidate Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"Total Candidates: {total_candidates} (Avg per S1: {avg_candidates:.1f})")

import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import re
import time
import pandas as pd
from business_entity_resolution.data.preprocessing import clean_business_name, normalize_text

print("Testing Blocking V3 (Controlled Candidate Size & High Recall)...")
gt_full = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', keep_default_na=False)
val_sample = gt_full.sample(n=2000, random_state=42).reset_index(drop=True)
val_s1_ids = set(val_sample['source1_entity_id'])

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
        return "", [], []
    norm = normalize_text(text)
    norm = re.sub(r'\b(h\s*no|hn|c\s*o|plot\s*no|flat\s*no|door\s*no|unit\s*unit|unit)\b', ' ', norm)
    tokens = norm.split()
    nums = [t for t in tokens if any(c.isdigit() for c in t)]
    words = [t for t in tokens if len(t) >= 4 and t not in COMMON_ADDR_WORDS and t not in STOPWORDS]
    return norm, nums, words

def get_blocking_keys_v3(country, name, addr):
    country = str(country).strip()
    clean_name, nt = clean_name_tokens(name)
    norm_addr, nums, words = get_address_info(addr)
    keys = set()
    
    # 1. Name keys
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
            
    # 2. Address keys: Number + Distinctive street word
    if nums and words:
        num = nums[0]
        for w in words[:2]:
            keys.add(f"AN:{country}:{num}_{w}")
            
    # Distinctive rare address word (length >= 6)
    for w in words[:3]:
        if len(w) >= 6:
            keys.add(f"AW:{country}:{w}")
            
    return keys

print("Building index...")
t1 = time.time()
block_index = {}
for eid, country, name, addr in zip(ref['entity_id'], ref['country'], ref['business_name'], ref['business_address']):
    b_keys = get_blocking_keys_v3(country, name, addr)
    for k in b_keys:
        if k not in block_index:
            block_index[k] = []
        block_index[k].append(eid)

print(f"Index built in {time.time()-t1:.2f}s with {len(block_index)} distinct keys.")

# Query S1 candidates with block size capping
MAX_BLOCK_SIZE = 500
t2 = time.time()
total_candidates = 0
found_matches = 0

for eid, country, name, addr in zip(val_s1['entity_id'], val_s1['country'], val_s1['business_name'], val_s1['business_address']):
    b_keys = get_blocking_keys_v3(country, name, addr)
    cands = set()
    for k in b_keys:
        lst = block_index.get(k)
        if lst and len(lst) <= MAX_BLOCK_SIZE:
            cands.update(lst)
    
    total_candidates += len(cands)
    true_set = gt_map.get(eid, set())
    found_matches += len(true_set & cands)

recall = found_matches / total_true_matches if total_true_matches > 0 else 0
avg_candidates = total_candidates / len(val_s1)
print(f"=== BLOCKING V3 RESULTS (MAX_BLOCK_SIZE={MAX_BLOCK_SIZE}) ===")
print(f"Total True Matches: {total_true_matches}")
print(f"Found Matches in Candidates: {found_matches}")
print(f"Candidate Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"Total Candidates: {total_candidates} (Avg per S1: {avg_candidates:.1f})")

import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import re
import time
import pandas as pd
from business_entity_resolution.data.preprocessing import clean_business_name, normalize_text

print("Loading test data for blocking v2 experiment...")
t0 = time.time()
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
print(f"Reference pool: {len(ref)} records ({len(s2)} S2, {len(s3)} S3). Loading took {time.time()-t0:.2f}s")

STOPWORDS = {"the", "and", "of", "in", "for", "a", "an", "on", "at", "to", "by", "with", "from", "service", "services", "center", "enterprises", "partners"}
COMMON_ADDR_WORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr", "court", "ct",
    "floor", "fl", "building", "bldg", "block", "blk", "sector", "sec", "phase", "plot",
    "nagar", "colony", "apartment", "apartments", "apt", "near", "opp", "opposite", "behind",
    "main", "cross", "north", "south", "east", "west", "india", "delhi", "bangalore", "mumbai"
}

def clean_name_v2(text):
    text = re.sub(r'[\*\#\>\<\[\]]', ' ', str(text))
    clean = clean_business_name(text)
    clean = re.sub(r'\.(com|org|net|in|co|io)\b', '', clean)
    tokens = [t for t in clean.split() if len(t) > 1 and t not in STOPWORDS]
    return tokens

def get_address_tokens_v2(text):
    if not text:
        return []
    norm = normalize_text(text)
    # Remove noise prefixes like hn, hno, c/o, plot no, flat no
    norm = re.sub(r'\b(h\s*no|hn|c\s*o|plot\s*no|flat\s*no|door\s*no|unit\s*unit|unit)\b', ' ', norm)
    tokens = [t for t in norm.split() if len(t) >= 4 and t not in COMMON_ADDR_WORDS and t not in STOPWORDS]
    return tokens

def get_blocking_keys_v2(country, name, addr):
    country = str(country).strip()
    keys = set()
    
    # Name keys
    nt = clean_name_v2(name)
    if nt:
        # First word
        w0 = nt[0]
        if len(w0) >= 3:
            keys.add(f"N:{country}:{w0}")
            # Prefix of length 5 for compound/domain words
            if len(w0) >= 6:
                keys.add(f"NP:{country}:{w0[:5]}")
        # Second word
        if len(nt) >= 2:
            w1 = nt[1]
            if len(w1) >= 4:
                keys.add(f"N:{country}:{w1}")
            keys.add(f"N2:{country}:{w0[:4]}_{w1[:4]}")
            
    # Address keys
    at = get_address_tokens_v2(addr)
    # Pick distinctive address tokens (length >= 5)
    for w in at[:4]:
        if len(w) >= 5:
            keys.add(f"A:{country}:{w}")
            
    # Number + address word
    norm_addr = normalize_text(addr)
    nums = [t for t in norm_addr.split() if any(c.isdigit() for c in t)]
    if nums and at:
        keys.add(f"AN:{country}:{nums[0]}_{at[0][:4]}")
        
    return keys

print("Building index with fast dictionary...")
t1 = time.time()
block_index = {}
# Fast iteration via zip
for eid, country, name, addr in zip(ref['entity_id'], ref['country'], ref['business_name'], ref['business_address']):
    b_keys = get_blocking_keys_v2(country, name, addr)
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

for eid, country, name, addr in zip(val_s1['entity_id'], val_s1['country'], val_s1['business_name'], val_s1['business_address']):
    b_keys = get_blocking_keys_v2(country, name, addr)
    cands = set()
    for k in b_keys:
        lst = block_index.get(k)
        if lst:
            # If block size is not huge (e.g. < 5000)
            if len(lst) < 5000:
                cands.update(lst)
    
    total_candidates += len(cands)
    true_set = gt_map.get(eid, set())
    found_matches += len(true_set & cands)

print(f"Querying took {time.time()-t2:.2f}s")
recall = found_matches / total_true_matches if total_true_matches > 0 else 0
avg_candidates = total_candidates / len(val_s1)
print(f"=== BLOCKING V2 RESULTS ===")
print(f"Total True Matches: {total_true_matches}")
print(f"Found Matches in Candidates: {found_matches}")
print(f"Candidate Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"Total Candidates: {total_candidates} (Avg per S1: {avg_candidates:.1f})")

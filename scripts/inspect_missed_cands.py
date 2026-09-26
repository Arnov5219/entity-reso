import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import re
import pandas as pd
from business_entity_resolution.data.preprocessing import clean_business_name, normalize_text

gt_full = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', keep_default_na=False)
val_sample = gt_full.sample(n=2000, random_state=42).reset_index(drop=True)

gt_map = {}
needed_matches = set()
for _, r in val_sample.iterrows():
    s1_id = r['source1_entity_id']
    m = r['matched_entity_ids']
    matches = set(x.strip() for x in m.split(',') if x.strip()) if m else set()
    gt_map[s1_id] = matches
    needed_matches.update(matches)

# Load S1
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', keep_default_na=False)
val_s1 = s1[s1['entity_id'].isin(set(val_sample['source1_entity_id']))].set_index('entity_id')

# Load needed S2/S3
records = {}
with open('dataset/train/train_source2.tsv', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in needed_matches:
            records[p[0]] = (p[1], p[2], p[3])

with open('dataset/train/train_source3.tsv', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in needed_matches:
            records[p[0]] = (p[1], p[2], p[3])

STOPWORDS = {"the", "and", "of", "in", "for", "a", "an", "on", "at", "to", "by", "with", "from"}

def get_name_tokens(text):
    clean = clean_business_name(text)
    clean = re.sub(r'\.(com|org|net|in|co|io)\b', '', clean)
    return [t for t in clean.split() if len(t) > 1 and t not in STOPWORDS]

def get_address_keys(text):
    if not text: return []
    norm = normalize_text(text)
    tokens = norm.split()
    nums = [t for t in tokens if any(c.isdigit() for c in t)]
    words = [t for t in tokens if t.isalpha() and len(t) > 2 and t not in STOPWORDS]
    keys = []
    if nums and words:
        num = nums[0]
        for w in words[:2]:
            keys.append(f"{num}_{w}")
    return keys

def get_keys(name, addr, country):
    keys = set()
    country = str(country).strip()
    nt = get_name_tokens(name)
    if nt:
        if len(nt[0]) >= 3:
            keys.add(f"N1:{country}:{nt[0]}")
        if len(nt) >= 2:
            keys.add(f"N2:{country}:{nt[0]}_{nt[1]}")
            if len(nt[1]) >= 4:
                keys.add(f"N1:{country}:{nt[1]}")
    for ak in get_address_keys(addr):
        keys.add(f"A:{country}:{ak}")
    return keys

missed = []
for s1_id, true_set in gt_map.items():
    if not true_set or s1_id not in val_s1.index:
        continue
    s1_row = val_s1.loc[s1_id]
    s1_keys = get_keys(s1_row['business_name'], s1_row['business_address'], s1_row['country'])
    
    for mid in true_set:
        if mid not in records: continue
        m_name, m_addr, m_country = records[mid]
        m_keys = get_keys(m_name, m_addr, m_country)
        if not (s1_keys & m_keys):
            missed.append((s1_id, s1_row['business_name'], s1_row['business_address'], mid, m_name, m_addr, s1_row['country']))

print(f"Total missed in sample: {len(missed)}")
print("\n" + "="*80)
for s1_id, s1_name, s1_addr, mid, m_name, m_addr, country in missed[:20]:
    print(f"S1: [{s1_id}] \"{s1_name}\" | Addr: \"{s1_addr}\" ({country})")
    print(f"  -> MISSED [{mid}] \"{m_name}\" | Addr: \"{m_addr}\"")
    print("-" * 80)

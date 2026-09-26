import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=25, keep_default_na=False)
s1_ids = set(gt['source1_entity_id'])
matched_ids = set()
for m in gt['matched_entity_ids']:
    if m:
        for x in m.split(','):
            matched_ids.add(x.strip())

s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', keep_default_na=False)
s1 = s1[s1['entity_id'].isin(s1_ids)].set_index('entity_id')

s2_matches = {x for x in matched_ids if x.startswith('S2-')}
s3_matches = {x for x in matched_ids if x.startswith('S3-')}

print(f'Looking for {len(s2_matches)} S2 matches and {len(s3_matches)} S3 matches...')
s2_records = {}
with open('dataset/train/train_source2.tsv', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s2_matches:
            s2_records[p[0]] = p[1:]
            if len(s2_records) == len(s2_matches):
                break

s3_records = {}
with open('dataset/train/train_source3.tsv', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s3_matches:
            s3_records[p[0]] = p[1:]
            if len(s3_records) == len(s3_matches):
                break

print('\n' + '=' * 80)
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    m_str = row['matched_entity_ids']
    if not m_str:
        print(f'S1: [{s1_id}] (SINGLETON - No true matches)')
        continue
    if s1_id not in s1.index:
        continue
    s1_row = s1.loc[s1_id]
    print(f'S1: [{s1_id}] Name: "{s1_row["business_name"]}" | Addr: "{s1_row["business_address"]}" | Country: {s1_row["country"]}')
    for mid in m_str.split(','):
        mid = mid.strip()
        if mid in s2_records:
            print(f'  -> S2: [{mid}] Name: "{s2_records[mid][0]}" | Addr: "{s2_records[mid][1]}"')
        elif mid in s3_records:
            print(f'  -> S3: [{mid}] Name: "{s3_records[mid][0]}" | Addr: "{s3_records[mid][1]}"')
    print('-' * 80)

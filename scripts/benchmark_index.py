import sys
sys.path.insert(0, 'src')
import time
import pandas as pd
from business_entity_resolution.blocking.blocker import EntityBlocker

blocker = EntityBlocker(max_block_size=2000)
t0 = time.time()
chunk = pd.read_csv('dataset/test/test_source2.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=500000)
print(f"Loaded 500k in {time.time()-t0:.2f}s")

t1 = time.time()
for eid, country, name, addr in zip(chunk['entity_id'], chunk['country'], chunk['business_name'], chunk['business_address']):
    blocker.index_record(eid, country, name, addr)
print(f"Indexed 500k in {time.time()-t1:.2f}s")

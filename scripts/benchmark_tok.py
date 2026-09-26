import time
import string

TRANS_PUNCT = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
s = '174 A, 7Th Floor, Tower A, The Corenthum Sector 62, Noida, Noida, Gautam Buddha Nagar, Uttar Pradesh'
t0 = time.time()
for _ in range(100000):
    norm = s.lower().translate(TRANS_PUNCT)
    tokens = norm.split()
print('100,000 strings tokenized in:', time.time()-t0, 'seconds!')

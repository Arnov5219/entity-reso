import string
from typing import Dict, List, Set, Tuple
import pandas as pd
from ..data.preprocessing import clean_business_name, normalize_text, US_STATES, IN_STATES

ALL_STATES = set(US_STATES.values()) | set(IN_STATES.values())
STOPWORDS = {
    "the", "and", "of", "in", "for", "a", "an", "on", "at", "to", "by", "with", "from",
    "service", "services", "center", "enterprises", "partners"
}
LEGAL_WORDS = {
    "llc", "inc", "corp", "corporation", "ltd", "limited", "pvt", "private", "co", "company",
    "llp", "gmbh", "sa", "sarl", "dba", "plc", "holdings", "group"
}
COMMON_ADDR_WORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr", "court", "ct",
    "floor", "fl", "building", "bldg", "block", "blk", "sector", "sec", "phase", "plot",
    "nagar", "colony", "apartment", "apartments", "apt", "near", "opp", "opposite", "behind",
    "main", "cross", "north", "south", "east", "west", "india", "delhi", "bangalore", "mumbai",
    "plaza", "tower", "complex", "house", "circle", "square", "park", "estate", "state", "city",
    "unit", "suite", "room", "first", "second", "third", "fourth", "fifth", "village", "post"
} | ALL_STATES

TRANS_PUNCT = str.maketrans(string.punctuation, ' ' * len(string.punctuation))


class EntityBlocker:
    """High-performance multi-pass blocking and candidate generator for Entity Resolution."""

    def __init__(self, max_block_size: int = 2000):
        self.max_block_size = max_block_size
        self.block_index: Dict[str, List[str]] = {}
        self.ref_meta: Dict[str, Tuple] = {}

    @staticmethod
    def get_blocking_keys_and_meta(country: str, name: str, addr: str):
        country = str(country).strip() if country else ""
        norm_name = str(name).lower().translate(TRANS_PUNCT) if name else ""
        norm_addr = str(addr).lower().translate(TRANS_PUNCT) if addr else ""

        name_raw_tokens = norm_name.split()
        nt = [t for t in name_raw_tokens if t not in LEGAL_WORDS and t not in STOPWORDS and len(t) > 1]
        clean_name = " ".join(nt)

        addr_raw_tokens = norm_addr.split()
        postals = set()
        nums = set()
        states = set()
        words = set()

        for t in addr_raw_tokens:
            if t.isdigit():
                nums.add(t)
                if len(t) in (5, 6):
                    postals.add(t)
            else:
                if t in US_STATES:
                    states.add(US_STATES[t])
                elif t in IN_STATES:
                    states.add(IN_STATES[t])
                elif t in ALL_STATES:
                    states.add(t)
                elif len(t) >= 4 and t not in COMMON_ADDR_WORDS and t not in STOPWORDS:
                    words.add(t)

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

    def index_record(self, eid: str, country: str, name: str, addr: str):
        """Directly index a single record without intermediate lists or DataFrames."""
        keys, meta = self.get_blocking_keys_and_meta(country, name, addr)
        self.ref_meta[eid] = meta
        for k in keys:
            if k not in self.block_index:
                self.block_index[k] = []
            self.block_index[k].append(eid)

    def fit_index(self, ref_df: pd.DataFrame):
        """Build index from candidate reference dataframe (S2 + S3)."""
        self.block_index = {}
        self.ref_meta = {}

        for eid, country, name, addr in zip(
            ref_df['entity_id'],
            ref_df['country'],
            ref_df['business_name'],
            ref_df['business_address']
        ):
            self.index_record(eid, country, name, addr)

    def get_candidates_for_record(self, country: str, name: str, addr: str) -> Set[str]:
        """Find candidate reference IDs for a single S1 record."""
        keys, _ = self.get_blocking_keys_and_meta(country, name, addr)
        candidates = set()
        for k in keys:
            c_list = self.block_index.get(k)
            if c_list and len(c_list) <= self.max_block_size:
                candidates.update(c_list)
        return candidates

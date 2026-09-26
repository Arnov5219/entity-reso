from typing import List, Tuple
from rapidfuzz import fuzz


class FeatureExtractor:
    """Extracts pairwise similarity and matching features for candidate entity pairs."""

    FEATURE_NAMES = [
        "n_ratio", "n_sort", "n_set", "n_jacc", "n_contain", "n_exact", "dom_m",
        "a_empty", "a_ratio", "a_sort", "a_set", "a_jacc", "a_contain",
        "num_match", "num_mismatch", "postal_match", "postal_mismatch",
        "state_match", "state_mismatch",
        "name_match_empty_addr", "addr_match_diff_name", "both_strong"
    ]

    @staticmethod
    def extract_features(s1_data: Tuple, r_data: Tuple) -> List[float]:
        s1_country, s1_name, s1_nt, s1_addr, s1_postals, s1_nums, s1_states, s1_words = s1_data
        r_country, r_name, r_nt, r_addr, r_postals, r_nums, r_states, r_words = r_data

        # Name similarities
        n_ratio = fuzz.ratio(s1_name, r_name) / 100.0 if (s1_name and r_name) else 0.0
        n_sort = fuzz.token_sort_ratio(s1_name, r_name) / 100.0 if (s1_name and r_name) else 0.0
        n_set = fuzz.token_set_ratio(s1_name, r_name) / 100.0 if (s1_name and r_name) else 0.0
        n_jacc = len(s1_nt & r_nt) / len(s1_nt | r_nt) if (s1_nt or r_nt) else 0.0
        n_contain = len(s1_nt & r_nt) / min(len(s1_nt), len(r_nt)) if (s1_nt and r_nt) else 0.0
        n_exact = 1.0 if (s1_name and r_name and s1_name == r_name) else 0.0

        # Domain / concatenated match
        s1_nospace = s1_name.replace(" ", "")
        r_nospace = r_name.replace(" ", "")
        dom_m = 1.0 if (s1_nospace and r_nospace and (s1_nospace in r_nospace or r_nospace in s1_nospace)) else 0.0

        # Address similarities
        a_empty = 1.0 if not r_addr else 0.0
        a_ratio = fuzz.ratio(s1_addr, r_addr) / 100.0 if (s1_addr and r_addr) else 0.0
        a_sort = fuzz.token_sort_ratio(s1_addr, r_addr) / 100.0 if (s1_addr and r_addr) else 0.0
        a_set = fuzz.token_set_ratio(s1_addr, r_addr) / 100.0 if (s1_addr and r_addr) else 0.0
        a_jacc = len(s1_words & r_words) / len(s1_words | r_words) if (s1_words or r_words) else 0.0
        a_contain = len(s1_words & r_words) / min(len(s1_words), len(r_words)) if (s1_words and r_words) else 0.0

        # Number logic
        if s1_nums and r_nums:
            shared_nums = s1_nums & r_nums
            num_match = 1.0 if bool(shared_nums) else 0.0
            num_mismatch = 1.0 if not bool(shared_nums) else 0.0
        else:
            num_match = 0.5
            num_mismatch = 0.0

        # Postal logic
        if s1_postals and r_postals:
            postal_match = 1.0 if bool(s1_postals & r_postals) else 0.0
            postal_mismatch = 1.0 if not bool(s1_postals & r_postals) else 0.0
        else:
            postal_match = 0.5
            postal_mismatch = 0.0

        # State logic
        if s1_states and r_states:
            state_match = 1.0 if bool(s1_states & r_states) else 0.0
            state_mismatch = 1.0 if not bool(s1_states & r_states) else 0.0
        else:
            state_match = 0.5
            state_mismatch = 0.0

        # Interaction features
        name_match_empty_addr = 1.0 if (a_empty == 1.0 and n_sort >= 0.85) else 0.0
        addr_match_diff_name = 1.0 if (a_set >= 0.80 and num_match == 1.0 and n_ratio < 0.50) else 0.0
        both_strong = 1.0 if (n_sort >= 0.75 and a_set >= 0.70 and num_mismatch == 0.0) else 0.0

        return [
            n_ratio, n_sort, n_set, n_jacc, n_contain, n_exact, dom_m,
            a_empty, a_ratio, a_sort, a_set, a_jacc, a_contain,
            num_match, num_mismatch, postal_match, postal_mismatch,
            state_match, state_mismatch,
            name_match_empty_addr, addr_match_diff_name, both_strong
        ]

import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from business_entity_resolution.blocking.blocker import EntityBlocker
from business_entity_resolution.features.extractor import FeatureExtractor


class TestFeatureExtraction(unittest.TestCase):
    def test_22_features(self):
        self.assertEqual(len(FeatureExtractor.FEATURE_NAMES), 22)

        blocker = EntityBlocker()
        _, s1_meta = blocker.get_blocking_keys_and_meta("US", "Zephay Labs Inc", "2621 Cotten Road, TX 75701")
        _, r_meta = blocker.get_blocking_keys_and_meta("US", "Zephay Labs Corp", "2621 Cotten Rd, TX 75701")

        feats = FeatureExtractor.extract_features(s1_meta, r_meta)
        self.assertEqual(len(feats), 22)
        # Check that numeric/postal matches are strong
        num_match_idx = FeatureExtractor.FEATURE_NAMES.index("num_match")
        self.assertEqual(feats[num_match_idx], 1.0)


if __name__ == "__main__":
    unittest.main()

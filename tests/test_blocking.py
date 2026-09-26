import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from business_entity_resolution.blocking.blocker import EntityBlocker


class TestBlocking(unittest.TestCase):
    def setUp(self):
        self.blocker = EntityBlocker(max_block_size=2000)

    def test_key_generation(self):
        keys, meta = self.blocker.get_blocking_keys_and_meta("US", "Zephay Labs Inc", "2621 Cotten Road, Tyler, TX 75701")
        self.assertTrue(any("zephay" in k for k in keys))
        self.assertTrue(any("cotten" in k for k in keys or "2621" in k))
        country, name, nt, addr, postals, nums, states, words = meta
        self.assertEqual(country, "US")
        self.assertIn("75701", postals)
        self.assertIn("2621", nums)

    def test_index_and_retrieve(self):
        self.blocker.index_record("S2-001", "US", "Zephay Labs Inc", "2621 Cotten Road, Tyler, TX 75701")
        cands = self.blocker.get_candidates_for_record("US", "Zephay Labs LLC", "2621 Cotten Rd, Tyler")
        self.assertIn("S2-001", cands)


if __name__ == "__main__":
    unittest.main()

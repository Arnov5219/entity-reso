import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from business_entity_resolution.data.preprocessing import (
    normalize_text,
    clean_business_name,
    normalize_address,
)


class TestPreprocessing(unittest.TestCase):
    def test_normalize_text(self):
        text = "  Acme, Inc. & Co.!  "
        norm = normalize_text(text)
        self.assertEqual(norm, "acme inc co")

    def test_clean_business_name(self):
        name = "Reliance Industries Pvt Ltd"
        clean = clean_business_name(name)
        self.assertEqual(clean, "reliance industries")

        us_name = "Global Logistics LLC"
        self.assertEqual(clean_business_name(us_name), "global logistics")

    def test_normalize_address(self):
        addr = "123 Main St, Apt 4B, TX"
        norm_addr = normalize_address(addr)
        self.assertIn("street", norm_addr)
        self.assertIn("apartment", norm_addr)
        self.assertIn("texas", norm_addr)


if __name__ == "__main__":
    unittest.main()

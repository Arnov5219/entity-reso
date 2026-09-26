import unittest
from pathlib import Path
from utils.validate_submission import MATCHING_HEADER, CANDIDATE_HEADER


class TestSubmissionValidation(unittest.TestCase):
    def test_headers(self):
        self.assertEqual(MATCHING_HEADER, ["source1_entity_id", "matched_entity_ids"])
        self.assertEqual(CANDIDATE_HEADER, ["source1_entity_id", "candidate_entity_ids"])

    def test_existing_output_files_exist(self):
        root = Path(__file__).resolve().parent.parent
        match_out = root / "output" / "matching_results.tsv"
        cand_out = root / "output" / "candidate_pairs.tsv"
        self.assertTrue(match_out.exists(), "matching_results.tsv missing")
        self.assertTrue(cand_out.exists(), "candidate_pairs.tsv missing")


if __name__ == "__main__":
    unittest.main()

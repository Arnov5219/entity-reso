import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from business_entity_resolution.models.matcher import EntityMatcher


class TestModels(unittest.TestCase):
    def test_load_production_model(self):
        model_path = Path("experiments/results/model.joblib")
        if not model_path.exists():
            model_path = Path("../experiments/results/model.joblib")
        self.assertTrue(model_path.exists(), f"Model file missing at {model_path}")

        matcher = EntityMatcher.load(model_path)
        self.assertEqual(matcher.threshold, 0.70)
        self.assertEqual(matcher.clf.__class__.__name__, "LGBMClassifier")
        self.assertEqual(matcher.clf.n_features_in_, 22)


if __name__ == "__main__":
    unittest.main()

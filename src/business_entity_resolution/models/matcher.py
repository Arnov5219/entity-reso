from pathlib import Path
from typing import Optional, Union
import joblib
import numpy as np
import lightgbm as lgb


class EntityMatcher:
    """GBDT-based pairwise entity matching classifier."""

    def __init__(
        self,
        model_type: str = "gbdt",
        n_estimators: int = 350,
        learning_rate: float = 0.06,
        num_leaves: int = 63,
        max_depth: int = 9,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        threshold: float = 0.70,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.threshold = threshold
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier on pairwise features."""
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict matching probability for pairs."""
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary match labels using threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def save(self, filepath: Union[str, Path]):
        """Save trained model to joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """Load trained model from joblib."""
        data = joblib.load(filepath)
        matcher = cls()
        if isinstance(data, dict) and "clf" in data:
            matcher.clf = data["clf"]
            matcher.threshold = data.get("threshold", 0.70)
        else:
            matcher.clf = data
            matcher.threshold = 0.70
        return matcher

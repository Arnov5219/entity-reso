from .pipeline import EntityResolutionPipeline, _to_gt_pairs
from .data.loader import DataLoader
from .data.preprocessing import preprocess_dataframe, normalize_text, clean_business_name, normalize_address
from .blocking.blocker import EntityBlocker
from .features.feature_engineering import FeatureExtractor
from .models.gbdt import EntityMatcher, GBDTMatcher
from .evaluation.metrics import evaluate_predictions, compute_entity_metrics, save_metrics, compare_to_baseline
from .utils import get_project_root

__all__ = [
    "EntityResolutionPipeline",
    "_to_gt_pairs",
    "DataLoader",
    "preprocess_dataframe",
    "normalize_text",
    "clean_business_name",
    "normalize_address",
    "EntityBlocker",
    "FeatureExtractor",
    "EntityMatcher",
    "GBDTMatcher",
    "evaluate_predictions",
    "compute_entity_metrics",
    "save_metrics",
    "compare_to_baseline",
    "get_project_root",
]

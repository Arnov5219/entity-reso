"""Unit tests for business_entity_resolution package."""
import sys
from pathlib import Path

# Add src to sys.path for test discovery
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

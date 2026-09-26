import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from business_entity_resolution.data.preprocessing import clean_business_name, normalize_text, US_STATES, IN_STATES
import lightgbm as lgb

# Let's inspect the 183 false negatives (true matches with prob < 0.65)
# and the 85 false positives (non-matches with prob >= 0.65)

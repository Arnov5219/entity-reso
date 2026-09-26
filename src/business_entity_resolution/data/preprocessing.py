import re
import unicodedata
from typing import Set, List
import pandas as pd

LEGAL_SUFFIXES = [
    r"\bprivate limited\b",
    r"\bpvt ltd\b",
    r"\bpvt\.?\s*ltd\.?\b",
    r"\bllc\b",
    r"\bl\.l\.c\.?\b",
    r"\binc\b",
    r"\bincorporated\b",
    r"\bcorp\b",
    r"\bcorporation\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\bllp\b",
    r"\bl\.l\.p\.?\b",
    r"\bltd\b",
    r"\blimited\b",
    r"\bplc\b",
    r"\bgmbh\b",
    r"\bsa\b",
    r"\bsarl\b",
    r"\bd\.?b\.?a\.?\b",
]
LEGAL_SUFFIX_REGEX = re.compile("|".join(LEGAL_SUFFIXES), flags=re.IGNORECASE)

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia"
}

IN_STATES = {
    "up": "uttar pradesh", "mp": "madhya pradesh", "ap": "andhra pradesh", "tn": "tamil nadu",
    "wb": "west bengal", "mh": "maharashtra", "gj": "gujarat", "rj": "rajasthan",
    "ka": "karnataka", "kl": "kerala", "dl": "delhi", "hr": "haryana", "pb": "punjab",
    "br": "bihar", "od": "odisha", "ts": "telangana", "uk": "uttarakhand", "jh": "jharkhand"
}

ADDRESS_ABBRS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "pkwy": "parkway",
    "hwy": "highway", "apt": "apartment", "ste": "suite", "fl": "floor", "bldg": "building",
    "pobox": "post office box", "nbr": "near", "opp": "opposite"
}


def normalize_text(text: str) -> str:
    """Basic normalization: Unicode NFKD, lowercase, accent removal, punctuation removal, whitespace strip."""
    if not text or pd.isna(text):
        return ""
    text = str(text)
    # NFKD normalization
    text = unicodedata.normalize("NFKD", text)
    # Lowercase
    text = text.lower()
    # Remove accents
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_business_name(text: str) -> str:
    """Normalize business name and strip legal suffixes."""
    norm = normalize_text(text)
    # Remove legal suffixes
    cleaned = LEGAL_SUFFIX_REGEX.sub(" ", norm)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else norm


def normalize_address(text: str) -> str:
    """Normalize address expanding common abbreviations."""
    norm = normalize_text(text)
    if not norm:
        return ""
    tokens = norm.split()
    expanded = []
    for t in tokens:
        if t in ADDRESS_ABBRS:
            expanded.append(ADDRESS_ABBRS[t])
        elif t in US_STATES:
            expanded.append(US_STATES[t])
        elif t in IN_STATES:
            expanded.append(IN_STATES[t])
        else:
            expanded.append(t)
    return " ".join(expanded)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns to a dataframe."""
    df = df.copy()
    if "business_name" in df.columns:
        df["name_norm"] = df["business_name"].fillna("").astype(str).map(normalize_text)
        df["name_clean"] = df["business_name"].fillna("").astype(str).map(clean_business_name)
    else:
        df["name_norm"] = ""
        df["name_clean"] = ""

    if "business_address" in df.columns:
        df["address_norm"] = df["business_address"].fillna("").astype(str).map(normalize_text)
        df["address_clean"] = df["business_address"].fillna("").astype(str).map(normalize_address)
    else:
        df["address_norm"] = ""
        df["address_clean"] = ""

    if "country" in df.columns:
        df["country_norm"] = df["country"].fillna("").astype(str).map(normalize_text)
    else:
        df["country_norm"] = ""

    return df

# Business Entity Resolution — Quick Guide

## The Problem
Match identical businesses across 3 different record sources, despite:
- Name variations ("ABC Pvt Ltd" vs "ABC Private Limited")
- Address formatting differences
- Typos, abbreviations, punctuation
- Missing data

**Challenge:** ~2.2M Source 1 records × ~5M Source 2 records = 11 trillion possible pairs → need smart filtering.

---

## The Solution (4 Stages)

### 1. **Preprocess** (Clean the data)
- Lowercase, remove punctuation
- Normalize addresses & remove legal suffixes (Ltd, Inc, etc.)
- Tokenize into words

### 2. **Block** (Filter to candidates only)
Try 3 strategies, measure which is best:
- **Strategy A:** Country + Name tokens
- **Strategy B:** Country + Address tokens  
- **Strategy C:** Country + (Name OR Address tokens)

**Critical:** Blocking quality = max possible recall. Missing true matches here = can't recover later.

### 3. **Match** (Score candidate pairs)
For each candidate pair, compute:
- Name similarity (exact match, token Jaccard, 3-gram)
- Address similarity (exact match, token Jaccard)
- Country match

**Simple scoring rule:**  
`Score = 0.6 × Name Similarity + 0.4 × Address Similarity`

### 4. **Evaluate** (Measure & decide)
- Sweep thresholds (0.5 to 0.9)
- Pick threshold that maximizes **F0.5** on validation data
- F0.5 = penalizes false matches more than missed matches

---

## Key Metrics

| Metric | Meaning |
|--------|---------|
| **Candidate Recall** | % of true matches that survive blocking |
| **Precision** | Of predicted matches, how many are correct? |
| **Recall** | Of all true matches, how many did we find? |
| **F0.5** | Weighted average (precision matters 2× more) |

---

## Team Roles (4 people)

| Role | Owns |
|------|------|
| **Team Lead** | Pipeline integration, AWS, final submission, experiment tracking |
| **Data Specialist** | EDA, data quality, missing values, edge cases |
| **Blocking Specialist** | Country/name/address filtering, candidate recall measurement |
| **ML Specialist** | Similarity features, scoring, threshold experiments, F0.5 calculation |

---

## 72-Hour Timeline

| Phase | What | Time |
|-------|------|------|
| 1 | Understand problem + setup | ~4h |
| 2 | EDA + baseline pipeline skeleton | ~12h |
| 3 | Test 3 blocking strategies | ~12h |
| 4 | Build features + threshold sweep | ~12h |
| 5 | Run on test data | ~4h |
| 6 | Validate + document + submit | ~4h |

---

## Success Criteria

✓ Reproducible pipeline  
✓ High candidate recall (blocking works)  
✓ Good validation F0.5  
✓ Reasonable runtime  
✓ Passes validation script  
✓ Clean documentation  

---

## Outputs Required

1. **matching_results.tsv** — Final predictions (scored)
2. **candidate_pairs.tsv** — Intermediate candidate set (for diagnosing blocking quality)

---

## Working Rules

- Measure everything (don't assume)
- Record every change + results
- Split validation by `source1_entity_id` (not random rows)
- Report progress: TASK → CHANGE → RESULT → METRICS
- Coordinate to avoid duplicate work

---

## What We're NOT Doing (Yet)

❌ Neural networks / embeddings  
❌ LLMs  
❌ Complex ensembles  
❌ External data enrichment  

→ **Baseline first. Improve with evidence.**

---

## Decision Tree (After Baseline)

```
Poor Candidate Recall?     → Improve blocking
Recall OK, F0.5 poor?      → Improve features
Poor Precision?            → Improve threshold
Poor Recall?               → Improve blocking/features
Runtime too high?          → Optimize indexing
```

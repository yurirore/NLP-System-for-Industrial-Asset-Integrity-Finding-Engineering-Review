# Model Card — Equipment Maintenance NLP Pipeline

## Model Details

### Overview

Three models compose the pipeline, each serving a distinct role:

| Model | Role | Type | Parameters |
|-------|------|------|------------|
| `all-MiniLM-L6-v2` | Sentence Embedding | Transformer (distilled BERT) | 22M |
| `RandomForestClassifier` | Recommendation Prediction | Ensemble (100 trees) | — |
| `google/flan-t5-small` (fine-tuned) | Action Sequence Generation | Encoder-Decoder Transformer | 80M |

### Embedding Model: all-MiniLM-L6-v2

- **Architecture:** 6-layer MiniLM distilled from BERT-base
- **Output dimension:** 384
- **Pooling:** Mean pooling (all tokens averaged)
- **Normalization:** Embeddings normalized to unit length
- **Training data:** 1B+ sentence pairs (NLI + web corpus)
- **Source:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **License:** Apache 2.0

### Classifier: Random Forest

- **Estimators:** 100 decision trees
- **Criterion:** Gini impurity
- **Features:** 384-dim sentence embeddings
- **Classes:** Multi-class maintenance recommendation labels
- **Training samples:** ~3,500 (70% of dataset)

### Summarization Model: FLAN-T5-Small (Fine-Tuned)

- **Base model:** `google/flan-t5-small`
- **Fine-tuning task:** Structured maintenance action generation
- **Training epochs:** 30 (best checkpoint at epoch 27)
- **Batch size:** 4 (CPU-constrained)
- **Optimizer:** Adam (default T5)
- **Warmup steps:** 500 (1.9% of total steps)
- **Weight decay:** 0.01
- **Input format:** `[TOPIC: ...] Finding: ... | Priority: ... | New recommendation: ... |`
- **Output format:** `[STEPS:N] Actions: 1. ... 2. ... 3. ...`
- **Generation parameters:**
  - Beam search width: 4
  - Max input length: 128 tokens
  - Max output length: 128 tokens
  - Early stopping: True

---

## Training Data

- **Source:** PETRONAS equipment inspection reports (anonymized)
- **Records:** ~5,000 maintenance findings
- **Splits:** 70/15/15 train/val/test (stratified by cluster)
- **Languages:** English (technical maintenance domain)
- **Preprocessing:**
  - Equipment code removal (regex: `[A-Z]-\d{4,}`)
  - Punctuation stripped
  - Lowercased
  - Stopwords kept (better for transformer embeddings)

### Topic Distribution

| Cluster | Label | Samples |
|---------|-------|---------|
| 0 | Valve leakage & insulation / alignment defects | ~823 |
| 1 | Structural steel perforation & severe corrosion | ~691 |
| 2 | Piping atmospheric corrosion & pitting | ~512 |
| 3 | Bolt & flange thick-scale corrosion | ~449 |
| 4 | Grating / studbolt corrosion & missing supports | ~738 |
| 5 | Bolt nut depletion & poor thread engagement | ~623 |
| 6 | Flange severe wall loss | ~564 |

---

## Evaluation Results

### Test Set Metrics

| Metric | Score | What It Measures |
|--------|-------|-----------------|
| **ROUGE-1** | 0.7456 | Unigram (word) overlap |
| **ROUGE-2** | 0.5821 | Bigram (phrase) overlap |
| **ROUGE-L** | **0.6283** | Longest common subsequence (action order) |
| **BERTScore F1** | **0.8924** | Semantic similarity (BERT embeddings) |
| **METEOR** | 0.6731 | Synonym-aware matching |

### Validation vs Test (Generalization Check)

| Metric | Validation | Test | Delta |
|--------|-----------|------|-------|
| ROUGE-L | 0.6391 | 0.6283 | +0.0108 |
| BERTScore F1 | 0.8967 | 0.8924 | +0.0043 |

**No overfitting detected** — test performance closely matches validation.

### Qualitative Examples

**Example 1 — Pump/Mechanical Failure**
```
Input:    centrifugal pump making loud noise and excessive vibration
Reference: Check alignment → Inspect bearings → Balance impeller → Tighten bolts
Prediction: Check pump alignment → Inspect bearing condition → Balance impeller if necessary → Secure mounting bolts
```
→ Semantically identical, valid wording variation.

**Example 2 — Leakage/Corrosion**
```
Input:    small leak observed at flange connection on pipeline
Reference: Isolate line → Inspect flange gasket → Replace gasket → Torque bolts → Test
Prediction: Shut off line → Check flange seal → Install new gasket → Tighten bolts to spec → Test for leaks
```
→ Perfect synonym usage, captured by BERTScore and METEOR.

---

## Intended Use

### Primary Use Case
- **Assistive tool** for maintenance engineers reviewing inspection findings
- Generates structured draft recommendations for human review
- Standardizes maintenance language across teams

### Out-of-Scope
- **Autonomous decision-making:** Always requires human approval
- **Safety-critical actions:** Final validation by qualified engineer
- **Domain transfer:** Trained on oil & gas corrosion/failure data only

### Limitations
- Fixed 7-topic taxonomy — does not discover new categories without retraining
- Rule-based actions are rigid (not learned)
- ~250ms latency acceptable for interactive use but not real-time

---

## Fairness & Bias

- **Domain-specific vocabulary:** Model performs best on oil & gas maintenance language
- **Imbalanced topics:** Clusters range from 449 to 823 samples — may affect minority topic accuracy
- **English only:** Not evaluated on multilingual maintenance reports

---

## Maintenance & Updates

- Recommended retraining cadence: **Quarterly**
- Monitor for topic drift via centroid distance tracking
- Low-confidence predictions (high centroid distance) should be flagged for human review
- User feedback (engineer corrections) can be collected for active learning

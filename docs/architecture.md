# Pipeline Architecture

## Overview

This pipeline processes equipment inspection findings and generates structured engineering recommendations. It combines three ML approaches — unsupervised clustering, supervised classification, and instruction-tuned sequence-to-sequence generation — with a rule-based safety layer.

---

## Data Flow

```
                        ┌──────────────────────────────┐
                        │   INPUT: Inspection Finding   │
                        │   {description, priority,     │
                        │    initial_recommendation}    │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   1. TEXT PREPROCESSING                              │
│                                                                     │
│   raw text ─► remove equipment codes (regex) ─► strip punctuation  │
│              ─► lowercase ─► cleaned text                           │
│                                                                     │
│   Purpose: Remove alphanumeric equipment IDs (P-1234, V-5678)       │
│   that would dominate the embedding space if left in.               │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────────┐
                    │              │                    │
                    ▼              ▼                    │
┌───────────────────────────┐  ┌───────────────────┐   │
│  2. EMBEDDING GENERATION  │  │  3. RF CLASSIFIER │   │
│                          │  │                    │   │
│  all-MiniLM-L6-v2       │  │  Random Forest     │   │
│  (SentenceTransformer)  │  │  Trained on        │   │
│  384-dim vector         │  │  combined_text     │   │
│  Mean pooling           │  │  (priority +       │   │
│  Normalize embeddings   │  │   description +    │   │
│                          │  │   initial rec)    │   │
└─────────────┬─────────────┘  └────────┬──────────┘   │
              │                         │               │
              ▼                         ▼               │
┌──────────────────────────┐            │               │
│  4. TOPIC CATEGORIZATION │            │               │
│                          │            │               │
│  Nearest Centroid        │            │               │
│  Euclidean distance      │            │               │
│  to 7 pre-computed       │            │               │
│  cluster centroids       │            │               │
│                          │            │               │
│  Output: defect topic    │            │               │
└─────────────┬────────────┘            │               │
              │                         │               │
              └─────────────┬───────────┘               │
                            │                           │
                            ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   5. FLAN-T5 SUMMARIZATION                          │
│                                                                     │
│   Input: [TOPIC: ...] Finding: ... | Priority: ... | New rec: ...  │
│                                                                     │
│   Model: google/flan-t5-small (80M params)                          │
│   Fine-tuned: 30 epochs, best @ epoch 27                            │
│   Generation: Beam search (k=4), max_length=128, early_stopping     │
│                                                                     │
│   Output: [STEPS:N] Actions: 1. ... 2. ... 3. ...                  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────────────┐
                        │   OUTPUT: Structured          │
                        │   Engineering Review          │
                        │   {recommendation, topic,    │
                        │    step-by-step actions}     │
                        └──────────────────────────────┘
```

---

## Design Decisions

### Why Hierarchical Clustering (not K-Means)?
- We didn't know the optimal number of topic clusters upfront
- Hierarchical clustering + dendrogram allowed visual exploration
- Ward's linkage creates balanced, interpretable clusters
- K=7 chosen based on **domain interpretability**, not just silhouette score

### Why FLAN-T5-Small (not Base/Large)?
| Variant | Params | CPU Inference | Fine-tune Time | Quality |
|---------|--------|---------------|----------------|---------|
| Small   | 80M    | ~200ms        | ~8 hours       | Good    |
| Base    | 250M   | ~600ms        | ~24 hours      | Better  |
| Large   | 780M   | ~2s           | ~72 hours      | Best    |

Small was chosen because it fits in 16GB RAM, trains in reasonable time on CPU, and achieves 89% BERTScore — sufficient for production.

### Why Rule-Based Actions + ML?
- **ML discovers patterns**: Clustering finds natural topic groupings
- **Rules ensure safety**: Oil & gas maintenance requires validated, compliant procedures
- **Hybrid**: Data-driven categorization + expert-validated action sequencing

### Why Offline Deployment?
- Corporate air-gapped environments (no internet)
- Models saved locally (`models/`, `nltk_data/`)
- No external API calls during inference

---

## Performance Characteristics

| Stage | Latency (CPU) | Memory |
|-------|--------------|--------|
| Text Cleaning | <1ms | — |
| Embedding | ~50ms | ~200MB |
| RF Classification | <1ms | ~50MB |
| Topic Matching | <1ms | ~1MB |
| T5 Generation | ~200ms | ~800MB |
| **Total** | **~250ms** | **~1.5GB** |

---

## Model Artifacts

| File | Size | Source |
|------|------|--------|
| `models/all-MiniLM-L6-v2/` | ~90MB | HuggingFace (offline copy) |
| `models/flan-t5-small/` | ~300MB | HuggingFace (offline copy) |
| `models/classifiers/random_forest_model.pkl` | ~3MB | Trained from dataset |
| `models/classifiers/label_encoder.pkl` | ~1KB | Trained from dataset |
| `models/classifiers/topic_centroids.pkl` | ~11KB | Computed from embeddings |
| `models/fine_tuned/summarization_model_joblib.pkl` | ~300MB | 30-epoch fine-tune |

See [reproduce_models.md](reproduce_models.md) for reproduction steps.

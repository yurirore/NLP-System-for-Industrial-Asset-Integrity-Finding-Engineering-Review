# Equipment Maintenance NLP Pipeline

A production-ready NLP pipeline for industrial equipment maintenance that processes inspection findings and generates structured engineering recommendations. Designed for **fully offline, CPU-only deployment** in corporate environments.

Built using unsupervised clustering (hierarchical), supervised classification (Random Forest), and fine-tuned instruction-tuned LLM (FLAN-T5-Small) — with a rule-based safety layer for compliance-critical action sequencing.

---

## Pipeline Architecture

```
                ┌──────────────────────┐
                │ Equipment Inspection │
                │ Finding Description  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Text Preprocessing  │
                │ (code removal, clean)│
                └──────────┬───────────┘
                           │
              ┌────────────┼────────────┐
              ▼                         ▼
    ┌─────────────────┐      ┌───────────────────┐
    │  Sentence Embed  │      │  Random Forest    │
    │  (MiniLM-L6-v2)  │      │  Classifier       │
    │  384-dim vector  │      │  (new rec label)  │
    └────────┬─────────┘      └────────┬──────────┘
             │                         │
             ▼                         │
    ┌──────────────────┐               │
    │  Nearest Centroid │              │
    │  (topic matching) │              │
    │  7 defect topics  │              │
    └────────┬─────────┘               │
             │                         │
             └──────┬───────────────-──┘
                    ▼
        ┌───────────────────────┐
        │   FLAN-T5-Small       │
        │  (fine-tuned 30 ep)   │
        │  Beam search (k=4)    │
        └──────────┬────────────┘
                   ▼
        ┌──────────────────────┐
        │  Structured Actions  │
        │  "1. Isolate pump"   │
        │  "2. Inspect seal"   │
        │  "3. Replace seal"   │
        └──────────────────────┘

```

## Tech Stack

| Component | Model / Library | Rationale |
|---|---|---|
| **Embeddings** | `all-MiniLM-L6-v2` (22M params) | Best speed/quality trade-off for CPU, 384-dim semantic embeddings |
| **Topic Discovery** | `AgglomerativeClustering` (Ward's linkage, K=7) | Hierarchical clustering — no need to pre-specify K; discovers natural maintenance categories |
| **Topic Inference** | Nearest Centroid | Fast inference: 7 distance comparisons vs re-running clustering |
| **Recommendation Classifier** | `RandomForestClassifier` | Multi-class prediction of maintenance action labels |
| **Summarization** | `google/flan-t5-small` (80M params, fine-tuned 30 epochs) | Instruction-tuned for structured output; CPU-inference in ~200ms |
| **Evaluation** | ROUGE-1/2/L, BERTScore (89.24% F1), METEOR | Semantic similarity + exact-match metrics |

## Performance

| Metric | Score | Interpretation |
|---|---|---|
| **ROUGE-L** | 62.83% | Strong sequential action capture |
| **ROUGE-2** | 58.21% | Good phrase-level matching |
| **BERTScore F1** | **89.24%** | Excellent semantic equivalence |
| **METEOR** | 67.31% | Handles domain synonyms well |
| **Inference Latency** | ~250ms per sample | Production-ready on CPU |

## 7 Maintenance Topics Discovered

The clustering algorithm automatically discovered these real-world defect categories:

| # | Topic | Sample Size |
|---|---|---|
| 0 | Valve leakage & insulation / alignment defects | ~823 |
| 1 | Structural steel perforation & severe corrosion | ~691 |
| 2 | Piping atmospheric corrosion & pitting | ~512 |
| 3 | Bolt & flange thick-scale corrosion | ~449 |
| 4 | Grating / studbolt corrosion & missing supports | ~738 |
| 5 | Bolt nut depletion & poor thread engagement | ~623 |
| 6 | Flange severe wall loss | ~564 |

## Project Structure

```
equipment-maintenance-pipeline/
│
├── src/                                  # Modular Python package
│   ├── pipeline.py                       # Main entry point
│   ├── preprocessing/
│   │   ├── text_cleaner.py               # Equipment code removal
│   │   └── nltk_setup.py                 # Offline NLTK config
│   ├── models/
│   │   ├── embedder.py                   # SentenceTransformer wrapper
│   │   ├── classifier.py                 # Random Forest wrapper
│   │   ├── topic_matcher.py              # Nearest-centroid topic inference
│   │   └── summarizer.py                 # FLAN-T5 inference wrapper
│   └── utils/
│       ├── config.py                     # Paths & constants
│       └── io_utils.py                   # CSV export helpers
│
├── notebooks/                            # Training & evaluation notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_clustering.ipynb
│   ├── 03_centroids_actions.ipynb
│   ├── 04_flan_t5_finetuning.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_inference_demo.ipynb
│
├── data/
│   ├── raw/                              # Original inspection data
│   ├── cleaned/                          # Processed datasets
│   └── samples/                          # Sample test batches
│
├── models/                               # Trained artifacts (see reproduce_models.md)
│   ├── all-MiniLM-L6-v2/                 # Sentence transformer (offline copy)
│   ├── flan-t5-small/                    # Pre-trained FLAN-T5 (offline copy)
│   ├── bert-base-uncased/                # For BERTScore evaluation
│   ├── classifiers/                      # RF model, label encoder, centroids
│   └── fine_tuned/                       # Fine-tuned T5 checkpoint
│
├── docs/                                 # Documentation
│   ├── interpretation.md                 # Full technical analysis of the training notebook
│   ├── architecture.md                   # Architecture deep-dive
│   ├── model_card.md                     # Model specifications & evaluation
│   └── reproduce_models.md               # Guide to reproduce trained models
│
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/equipment-maintenance-pipeline.git
cd equipment-maintenance-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the inference pipeline (single sample)
python -m src.pipeline

# 4. Or use programmatically
from src.pipeline import load_all_models, run_complete_pipeline

models = load_all_models()
result = run_complete_pipeline(
    description="pump seal leaking, oil puddle observed under pump",
    priority="2",
    initial_recommendation="to be rectified",
    verbose=True,
    **models,
)
print(result['structured_output'])
# → Isolate pump → Drain oil → Inspect seal → Replace → Refill → Test
```

## Reproducing the Models

See [`docs/reproduce_models.md`](docs/reproduce_models.md) for step-by-step instructions to:
- Download pre-trained models
- Fine-tune FLAN-T5-Small from scratch (30 epochs, ~8 hours on CPU)
- Re-run hierarchical clustering
- Reproduce evaluation metrics

## License

MIT — see [LICENSE](LICENSE).

---

*My internship project built for PETRONAS equipment maintenance. Designed to augment, not replace, engineering expertise.*

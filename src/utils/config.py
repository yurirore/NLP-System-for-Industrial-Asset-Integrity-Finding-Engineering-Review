"""
Project-wide configuration: paths, constants, and model file locations.

All paths resolve relative to the project root (parent of src/),
making the pipeline portable across machines.
"""

import os

# ── Project root resolution ──────────────────────────────────────────────
# src/utils/config.py → climb up 3 levels to reach project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Key directories ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLASSIFIER_DIR = os.path.join(MODEL_DIR, "classifiers")
FINE_TUNED_DIR = os.path.join(MODEL_DIR, "fine_tuned")
MINILM_DIR = os.path.join(MODEL_DIR, "all-MiniLM-L6-v2")
CENTROIDS_PATH = os.path.join(CLASSIFIER_DIR, "topic_centroids.pkl")
SUMMARIZATION_PATH = os.path.join(FINE_TUNED_DIR, "summarization_model_joblib.pkl")

NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

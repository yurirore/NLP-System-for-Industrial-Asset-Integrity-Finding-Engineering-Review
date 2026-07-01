"""
Topic matching module using nearest-centroid classification on sentence embeddings.
"""

import os
import numpy as np
import joblib


# 7 maintenance defect categories discovered via hierarchical clustering
TOPIC_TAGS = {
    0: "valve leakage & insulation / alignment defects",
    1: "structural steel perforation & severe corrosion",
    2: "piping atmospheric corrosion & pitting",
    3: "bolt & flange thick-scale corrosion",
    4: "grating / studbolt corrosion & missing supports",
    5: "bolt nut depletion & poor thread engagement",
    6: "flange severe wall loss",
}


def load_centroids(centroids_path: str) -> dict:
    """
    Load pre-computed cluster centroids for topic inference.

    Args:
        centroids_path: Path to the centroids pickle file.

    Returns:
        dict: Mapping of cluster_id (int) → centroid vector (np.ndarray).

    Raises:
        FileNotFoundError: If the centroids file is missing.
    """
    if not os.path.exists(centroids_path):
        raise FileNotFoundError(f"❌ topic_centroids.pkl not found at: {centroids_path}")

    centroids = joblib.load(centroids_path)
    print(f"   ✅ Centroids loaded from: {centroids_path}")
    return centroids


def infer_topic(description_cleaned: str, embedder, centroids: dict) -> str:
    """
    Infer equipment defect topic using nearest-centroid classification.

    Args:
        description_cleaned: Cleaned equipment description (lowercased, no codes).
        embedder: SentenceTransformer model for generating embeddings.
        centroids: Pre-computed cluster centroids (dict of id → vector).

    Returns:
        str: Equipment defect category label.
    """
    # Generate embedding for the cleaned description
    embedding = embedder.encode([description_cleaned], normalize_embeddings=True)[0]

    # Find nearest centroid using Euclidean distance
    distances = {
        cid: np.linalg.norm(embedding - centroid)
        for cid, centroid in centroids.items()
    }
    cluster_id = min(distances, key=distances.get)

    # Map to topic tag
    return TOPIC_TAGS.get(cluster_id, f"Unknown Cluster ({cluster_id})")

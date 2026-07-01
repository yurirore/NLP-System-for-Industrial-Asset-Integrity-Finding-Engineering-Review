"""
Sentence embedding model wrapper.

Handles loading all-MiniLM-L6-v2 as a SentenceTransformer model
and generating embeddings for equipment descriptions.
"""

import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling


def load_embedder(minilm_dir: str) -> SentenceTransformer:
    """
    Load the all-MiniLM-L6-v2 SentenceTransformer model.

    The model is loaded by manually assembling Transformer + Pooling modules
    (rather than loading via model name) to support fully offline operation
    without depending on the HuggingFace cache.

    Args:
        minilm_dir: Path to the all-MiniLM-L6-v2 model directory.

    Returns:
        Configured SentenceTransformer model.

    Raises:
        FileNotFoundError: If the Transformer subdirectory is missing.
        RuntimeError: If model loading fails.
    """
    transformer_dir = os.path.join(minilm_dir, "0_Transformer")
    pooling_dir = os.path.join(minilm_dir, "1_Pooling")

    if not os.path.exists(transformer_dir):
        raise FileNotFoundError(
            f"Transformer components not found at: {transformer_dir}\n"
            f"Expected structure: {minilm_dir}/0_Transformer/"
        )

    try:
        word_embedding_model = Transformer(transformer_dir)
        pooling_model = Pooling(
            embedding_dimension=word_embedding_model.get_embedding_dimension(),
            pooling_mode='mean',
        )
        embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print(f"   ✅ Sentence Transformer loaded from: {minilm_dir}")
        return embedder
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load Sentence Transformer: {e}")

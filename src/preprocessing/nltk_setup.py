"""
NLTK data path configuration for offline/corporate environments.
"""

import os
import nltk


def setup_nltk(nltk_data_dir: str = None) -> None:
    """
    Configure NLTK to look for data in a local directory.

    This allows the pipeline to run fully offline in corporate/air-gapped
    environments without depending on the default NLTK download paths.

    Args:
        nltk_data_dir: Path to local NLTK data directory.
                       Defaults to './nltk_data' relative to the project root.
    """
    if nltk_data_dir is None:
        # Resolve relative to this file's location (src/preprocessing/)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        nltk_data_dir = os.path.join(base_dir, "nltk_data")

    if os.path.exists(nltk_data_dir):
        nltk.data.path.append(nltk_data_dir)
        print(f"📦 NLTK data path set to: {nltk_data_dir}")
    else:
        print(f"⚠️  NLTK data directory not found at: {nltk_data_dir}")
        print("   Download will fall back to default NLTK paths.")

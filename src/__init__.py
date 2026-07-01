"""
Industrial Equipment Maintenance NLP Pipeline.

A complete pipeline for processing equipment inspection findings and generating
structured engineering recommendations using multiple ML models.

Pipeline Stages:
1. Embedding: Sentence embedding generation (all-MiniLM-L6-v2)
2. Classification: Predict new recommendation label (Random Forest)
3. Topic Matching: Nearest-centroid defect category inference
4. Summarization: Structured action steps generation (FLAN-T5-Small)
"""

__version__ = "0.1.0"

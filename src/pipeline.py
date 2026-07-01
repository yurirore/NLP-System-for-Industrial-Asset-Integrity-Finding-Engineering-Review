"""
Industrial Equipment Maintenance Pipeline
==========================================

A complete inference pipeline for processing equipment inspection findings
and generating structured engineering recommendations.

Pipeline Stages:
    1. Embedding Generation — SentenceTransformer (all-MiniLM-L6-v2)
    2. Classification — Random Forest predicts new recommendation label
    3. Topic Categorization — Nearest-centroid defect category inference
    4. Summarization — Fine-tuned FLAN-T5 generates structured actions

Usage:
    python -m src.pipeline
"""

import os
import sys

# Allow running as `python -m src.pipeline` or `python src/pipeline.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.preprocessing.text_cleaner import clean_description
from src.preprocessing.nltk_setup import setup_nltk
from src.models.embedder import load_embedder
from src.models.classifier import load_classifier, predict_recommendation
from src.models.topic_matcher import load_centroids, infer_topic
from src.models.summarizer import load_summarizer, generate_structured_output
from src.utils.config import (
    MODEL_DIR, CLASSIFIER_DIR, MINILM_DIR, CENTROIDS_PATH,
    SUMMARIZATION_PATH, NLTK_DIR, SAMPLE_DATA_DIR, OUTPUT_DIR,
)
from src.utils.io_utils import save_results_to_csv


# =========================================================================
# 1. MODEL LOADING
# =========================================================================

def load_all_models():
    """
    Load all required models and components for the pipeline.

    Returns:
        tuple: (classification_model, label_encoder, summarization_model,
                summarization_tokenizer, embedder, centroids)
    """
    print(f"\n📂 Model directory: {MODEL_DIR}")

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"\n❌ Models directory not found!\n   Expected: {MODEL_DIR}\n"
            f"   Please ensure models are set up in the 'models/' folder."
        )

    # [1/4] Classification Model
    print("\n[1/4] Loading Classification Model...")
    classification_model, label_encoder = load_classifier(CLASSIFIER_DIR)

    # [2/4] Summarization Model
    print("\n[2/4] Loading Summarization Model...")
    summarization_model, summarization_tokenizer = load_summarizer(SUMMARIZATION_PATH)

    # [3/4] Sentence Transformer
    print("\n[3/4] Loading Sentence Transformer for embeddings...")
    embedder = load_embedder(MINILM_DIR)

    # [4/4] Cluster Centroids
    print("\n[4/4] Loading cluster centroids...")
    centroids = load_centroids(CENTROIDS_PATH)

    print("\n" + "=" * 50)
    print("✅ All models loaded successfully!")
    print("=" * 50)

    return (
        classification_model, label_encoder,
        summarization_model, summarization_tokenizer,
        embedder, centroids,
    )


# =========================================================================
# 2. MAIN PIPELINE
# =========================================================================

def run_complete_pipeline(
    description,
    priority,
    initial_recommendation,
    classification_model,
    label_encoder,
    summarization_model,
    summarization_tokenizer,
    embedder,
    centroids,
    verbose=True,
):
    """
    Complete pipeline: raw input → structured maintenance recommendation.

    Args:
        description: Finding description text.
        priority: Priority level (e.g., '1', '2', '3').
        initial_recommendation: Initial recommendation string.
        classification_model: Trained Random Forest classifier.
        label_encoder: Label encoder for recommendation classes.
        summarization_model: Fine-tuned T5 model.
        summarization_tokenizer: T5 tokenizer.
        embedder: Sentence transformer for embeddings.
        centroids: Cluster centroids for topic inference.
        verbose: Whether to print detailed progress.

    Returns:
        dict: Contains original_input, predicted_recommendation,
              topic_category, and structured_output.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STARTING PIPELINE")
        print("=" * 70)

    # ── Step 1: Predict NEW RECOMMENDATION ────────────────────────────
    if verbose:
        print("\n[STEP 1] Classification — Predicting NEW RECOMMENDATION")
        print("-" * 70)

    combined_text = (
        f"Priority: {priority}. "
        f"Description: {description}. "
        f"Initial Rec: {initial_recommendation}. "
    )
    if verbose:
        print(f"Input: {combined_text}")

    embedding = embedder.encode([combined_text])
    predicted_recommendation = predict_recommendation(
        embedding, classification_model, label_encoder
    )

    if verbose:
        print(f"\n✅ Predicted NEW RECOMMENDATION: '{predicted_recommendation}'")

    # ── Step 2: Clean description & infer topic ───────────────────────
    if verbose:
        print("\n[STEP 2] Topic Categorization")
        print("-" * 70)

    description_cleaned = clean_description(description)
    topic_category = infer_topic(description_cleaned, embedder, centroids)

    if verbose:
        print(f"   Cleaned Description: {description_cleaned}")
        print(f"   Topic Category: {topic_category}")

    # ── Step 3: Generate structured output ────────────────────────────
    if verbose:
        print("\n[STEP 3] Summarization — Generating Structured Output")
        print("-" * 70)

    summarization_input = (
        f"[TOPIC: {topic_category}] "
        f"Finding: {description_cleaned} | "
        f"Priority: {priority} | "
        f"New recommendation: {predicted_recommendation} | "
    )

    if verbose:
        print(f"Input to Summarization Model:\n{summarization_input}")

    structured_output = generate_structured_output(
        summarization_model,
        summarization_tokenizer,
        summarization_input,
    )

    if verbose:
        print(f"\n✅ Generated Structured Output:\n{structured_output}")
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED")
        print("=" * 70)

    return {
        'original_input': {
            'description': description,
            'priority': priority,
            'initial_recommendation': initial_recommendation,
        },
        'predicted_recommendation': predicted_recommendation,
        'topic_category': topic_category,
        'structured_output': structured_output,
    }


# =========================================================================
# 3. BATCH PROCESSING
# =========================================================================

def batch_process(input_df, **model_kwargs):
    """
    Process multiple inputs from a DataFrame.

    Args:
        input_df: DataFrame with columns DESCRIPTION, PRIORITY,
                  INITIAL RECOMMENDATION.
        **model_kwargs: All loaded model objects passed to run_complete_pipeline.

    Returns:
        list: List of result dictionaries.
    """
    results = []
    total = len(input_df)

    for idx, row in input_df.iterrows():
        print(f"Processing {idx + 1}/{total}...", end="\r")

        result = run_complete_pipeline(
            description=row['DESCRIPTION'],
            priority=str(row['PRIORITY']),
            initial_recommendation=row['INITIAL RECOMMENDATION'],
            verbose=False,
            **model_kwargs,
        )
        results.append(result)

    print(f"\n✅ Batch processing complete! Processed {total} items.")
    return results


# =========================================================================
# 4. MAIN ENTRY POINT
# =========================================================================

def main():
    """Main execution — demonstrates the complete pipeline workflow."""
    print("=" * 70)
    print("INDUSTRIAL EQUIPMENT MAINTENANCE PIPELINE")
    print("=" * 70)

    # Load all models
    print("\n[INITIALIZATION] Loading models...")
    models = load_all_models()

    model_kwargs = {
        'classification_model': models[0],
        'label_encoder': models[1],
        'summarization_model': models[2],
        'summarization_tokenizer': models[3],
        'embedder': models[4],
        'centroids': models[5],
    }

    # ── Test 1: Single sample inference ──────────────────────────────
    print("\n\n" + "=" * 70)
    print("TEST 1: SINGLE SAMPLE INFERENCE")
    print("=" * 70)

    result = run_complete_pipeline(
        description="direct contact with pipe support with crevice corrosion",
        priority="2",
        initial_recommendation="to be rectified",
        verbose=True,
        **model_kwargs,
    )

    print("\n\n" + "#" * 70)
    print("SINGLE SAMPLE FINAL RESULTS")
    print("#" * 70)
    print(f"\nPredicted Recommendation: {result['predicted_recommendation']}")
    print(f"\nStructured Engineering Review Output:\n{result['structured_output']}")

    # ── Test 2: Batch processing ────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("TEST 2: BATCH PROCESSING")
    print("=" * 70)

    sample_file = os.path.join(SAMPLE_DATA_DIR, "test_batch.csv")
    if os.path.exists(sample_file):
        sample_batch = pd.read_csv(sample_file)
        print(f"\nLoaded batch with {len(sample_batch)} samples")
        print("Starting batch processing...\n")

        batch_results = batch_process(sample_batch, **model_kwargs)

        print("\nSaving results...")
        output_path = os.path.join(OUTPUT_DIR, "pipeline_results.csv")
        df_results = save_results_to_csv(batch_results, output_path)

        print(f"\n✅ Pipeline execution complete!")
        print(f"   Processed {len(df_results)} samples successfully")
        print(f"\nSample output (first 3 rows):")
        print(df_results.head(3).to_string())
    else:
        print(f"⚠️  Sample batch file not found at: {sample_file}")
        print("   Skipping batch processing test.")


if __name__ == "__main__":
    main()

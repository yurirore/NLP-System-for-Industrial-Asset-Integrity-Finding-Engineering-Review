"""
Unit tests for pipeline orchestration and configuration.
"""

import pytest
from src.utils.config import (
    BASE_DIR, MODEL_DIR, CLASSIFIER_DIR, FINE_TUNED_DIR,
    CENTROIDS_PATH, SUMMARIZATION_PATH,
)


class TestConfigPaths:
    """Verify that all configuration paths resolve correctly."""

    def test_base_dir_exists(self):
        """BASE_DIR should point to the project root."""
        assert BASE_DIR is not None
        assert BASE_DIR.endswith("helparooni")

    def test_model_dir_resolves(self):
        """MODEL_DIR should point to the models directory."""
        assert "models" in MODEL_DIR

    def test_classifier_dir_resolves(self):
        """CLASSIFIER_DIR should end with classifiers."""
        assert CLASSIFIER_DIR.endswith("classifiers")

    def test_fine_tuned_dir_resolves(self):
        """FINE_TUNED_DIR should end with fine_tuned."""
        assert FINE_TUNED_DIR.endswith("fine_tuned")

    def test_centroids_path_resolves(self):
        """CENTROIDS_PATH should point to the centroids pickle."""
        assert CENTROIDS_PATH.endswith("topic_centroids.pkl")

    def test_summarization_path_resolves(self):
        """SUMMARIZATION_PATH should point to the summarization model."""
        assert SUMMARIZATION_PATH.endswith("summarization_model_joblib.pkl")


class TestModelLoading:
    """Test that model loading works end-to-end.

    These tests require the trained model artifacts to be present.
    Skip if models aren't available (e.g., in CI without artifacts).
    """

    def test_load_all_models(self):
        """load_all_models should load all 6 components."""
        try:
            from src.pipeline import load_all_models
            models = load_all_models()
            assert len(models) == 6
            names = [type(m).__name__ for m in models]
            assert "RandomForestClassifier" in names
            assert "LabelEncoder" in names
            assert "T5ForConditionalGeneration" in names
            assert "T5Tokenizer" in names
            assert "SentenceTransformer" in names
            assert "dict" in names  # centroids
        except (FileNotFoundError, RuntimeError) as e:
            pytest.skip(f"Model artifacts not available: {e}")

    def test_single_sample_inference(self):
        """run_complete_pipeline should return structured output."""
        try:
            from src.pipeline import load_all_models, run_complete_pipeline
            models = load_all_models()
            result = run_complete_pipeline(
                description="ubolt missing",
                priority="4",
                initial_recommendation="to be rectified",
                classification_model=models[0],
                label_encoder=models[1],
                summarization_model=models[2],
                summarization_tokenizer=models[3],
                embedder=models[4],
                centroids=models[5],
                verbose=False,
            )
            assert "original_input" in result
            assert "predicted_recommendation" in result
            assert "topic_category" in result
            assert "structured_output" in result
            assert isinstance(result["structured_output"], str)
            assert len(result["structured_output"]) > 0
        except (FileNotFoundError, RuntimeError) as e:
            pytest.skip(f"Model artifacts not available: {e}")


class TestBatchProcessing:
    """Test batch processing with sample data."""

    def test_batch_process_with_csv(self, sample_batch_csv):
        """batch_process should handle a CSV with multiple rows."""
        try:
            import pandas as pd
            from src.pipeline import load_all_models, batch_process

            models = load_all_models()
            df = pd.read_csv(sample_batch_csv)

            results = batch_process(
                df,
                classification_model=models[0],
                label_encoder=models[1],
                summarization_model=models[2],
                summarization_tokenizer=models[3],
                embedder=models[4],
                centroids=models[5],
            )
            assert len(results) == len(df)
            for r in results:
                assert "structured_output" in r
        except (FileNotFoundError, RuntimeError) as e:
            pytest.skip(f"Model artifacts not available: {e}")


class TestIOUtilities:
    """Test the CSV export utility."""

    def test_save_results_creates_csv(self, tmp_path):
        """save_results_to_csv should create a valid CSV file."""
        from src.utils.io_utils import save_results_to_csv

        results = [
            {
                'original_input': {
                    'description': 'test leak',
                    'priority': '2',
                    'initial_recommendation': 'fix',
                },
                'predicted_recommendation': 'repair',
                'topic_category': 'Leakage/Corrosion',
                'structured_output': 'Isolate → Inspect → Repair',
            }
        ]

        output_path = tmp_path / "test_output.csv"
        df = save_results_to_csv(results, str(output_path))

        assert output_path.exists()
        assert len(df) == 1
        assert df.iloc[0]['PREDICTED_RECOMMENDATION'] == 'repair'

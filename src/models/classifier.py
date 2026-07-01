"""
Random Forest classifier wrapper for maintenance recommendation prediction.
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def load_classifier(classifier_dir: str):
    """
    Load the trained Random Forest model and its label encoder.

    Args:
        classifier_dir: Path to the classifiers directory.

    Returns:
        tuple: (classification_model, label_encoder)

    Raises:
        FileNotFoundError: If model or encoder files are missing.
    """
    rf_path = os.path.join(classifier_dir, "random_forest_model.pkl")
    le_path = os.path.join(classifier_dir, "label_encoder.pkl")

    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"❌ random_forest_model.pkl not found at: {rf_path}")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"❌ label_encoder.pkl not found at: {le_path}")

    classification_model = joblib.load(rf_path)
    label_encoder = joblib.load(le_path)

    print(f"   ✅ Random Forest loaded from: {rf_path}")
    print(f"   ✅ Label Encoder loaded from: {le_path}")

    return classification_model, label_encoder


def predict_recommendation(embedding, classification_model, label_encoder) -> str:
    """
    Predict a new maintenance recommendation from an embedding.

    Args:
        embedding: Sentence embedding vector.
        classification_model: Trained Random Forest classifier.
        label_encoder: Label encoder for recommendation classes.

    Returns:
        str: Predicted recommendation label.
    """
    prediction_numeric = classification_model.predict(embedding)
    return label_encoder.inverse_transform(prediction_numeric)[0]

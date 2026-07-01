"""
FLAN-T5 summarization model wrapper for generating structured maintenance actions.
"""

import os
import pickle
import joblib
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


def _patch_tokenizer_setstate():
    """
    Monkey-patch T5Tokenizer.__setstate__ to handle cross-platform path issues.

    The summarization model was pickled on Windows, and the T5 tokenizer stores
    an absolute path to spiece.model in its state. When unpickled on another OS,
    this path doesn't exist and the unpickling fails.

    This patch catches the Load error during __setstate__ and lets the tokenizer
    load without the spiece model, which we then fix after unpickling.
    """
    original_setstate = T5Tokenizer.__setstate__

    def patched_setstate(self, state):
        try:
            original_setstate(self, state)
        except Exception:
            # Load state dict manually, skipping the sp_model.Load call
            vocab_file = state.get('vocab_file', '')
            self.vocab_file = vocab_file
            # Initialize sp_model from the state's serialized model if available
            if 'sp_model' in state:
                self.sp_model = state['sp_model']
            else:
                from sentencepiece import SentencePieceProcessor
                self.sp_model = SentencePieceProcessor()

    T5Tokenizer.__setstate__ = patched_setstate


def _restore_tokenizer_setstate():
    """Restore original T5Tokenizer.__setstate__ to avoid side effects."""
    if hasattr(T5Tokenizer, '_original_setstate'):
        T5Tokenizer.__setstate__ = T5Tokenizer._original_setstate


def load_summarizer(summary_path: str, fallback_pretrained_dir: str = None):
    """
    Load the fine-tuned FLAN-T5 model and tokenizer from a combined joblib package.

    The pickle was created on Windows and contains hardcoded paths for the
    sentencepiece model. This loader patches the tokenizer to handle the
    cross-platform path mismatch gracefully.

    Args:
        summary_path: Path to the summarization_model_joblib.pkl file.
        fallback_pretrained_dir: Path to the flan-t5-small directory containing
                                 spiece.model. Defaults to models/flan-t5-small
                                 relative to the fine_tuned directory.

    Returns:
        tuple: (model, tokenizer)

    Raises:
        FileNotFoundError: If the model file is missing.
        ValueError: If the package is missing expected keys.
    """
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"❌ summarization_model_joblib.pkl not found at: {summary_path}")

    # Determine fallback spiece.model path
    if fallback_pretrained_dir is None:
        fallback_pretrained_dir = os.path.join(
            os.path.dirname(os.path.dirname(summary_path)), "flan-t5-small"
        )
    spiece_path = os.path.join(fallback_pretrained_dir, "spiece.model")

    # Patch tokenizer to survive cross-platform unpickling
    _patch_tokenizer_setstate()

    try:
        summarization_package = joblib.load(summary_path)
    finally:
        _restore_tokenizer_setstate()

    model = summarization_package.get('model')
    tokenizer = summarization_package.get('tokenizer')

    if model is None or tokenizer is None:
        raise ValueError(
            "❌ summarization_model_joblib.pkl is corrupted. "
            "Expected 'model' and 'tokenizer' keys."
        )

    # Fix the tokenizer's sentencepiece model path
    if os.path.exists(spiece_path):
        try:
            tokenizer.sp_model.Load(spiece_path)
            tokenizer.vocab_file = spiece_path
        except Exception:
            print(f"   ⚠️  Could not load spiece.model at: {spiece_path}")

    # Verify the tokenizer works; if not, reload from pretrained
    try:
        tokenizer("test", return_tensors="pt")
    except Exception:
        print("   ⚠️  Tokenizer state invalid, reloading from pretrained...")
        tokenizer = T5Tokenizer.from_pretrained(fallback_pretrained_dir)

    print(f"   ✅ T5 Model loaded from: {summary_path}")
    return model, tokenizer


def generate_structured_output(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    input_text: str,
    max_input_length: int = 128,
    max_output_length: int = 128,
    num_beams: int = 4,
) -> str:
    """
    Generate a structured maintenance recommendation from a formatted input string.

    Args:
        model: Fine-tuned FLAN-T5 model.
        tokenizer: T5 tokenizer.
        input_text: Formatted input string (topic, description, etc.).
        max_input_length: Maximum token length for input truncation.
        max_output_length: Maximum token length for generation.
        num_beams: Beam search width. Higher = better quality, slower.

    Returns:
        str: Generated structured maintenance output.
    """
    inputs = tokenizer(
        [input_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

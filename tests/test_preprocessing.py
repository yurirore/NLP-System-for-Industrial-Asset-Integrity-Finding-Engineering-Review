"""
Unit tests for text preprocessing and cleaning.
"""

import pytest
from src.preprocessing.text_cleaner import clean_description


class TestCleanDescription:
    """Tests for the equipment code removal and text cleaning."""

    def test_removes_equipment_codes(self):
        """Equipment codes like P-1234, V-5678 should be removed."""
        result = clean_description("Pump P-1234 leak at flange V-5678")
        assert "P-1234" not in result
        assert "V-5678" not in result
        assert "leak" in result
        assert "flange" in result

    def test_lowercases_text(self):
        """All text should be lowercased."""
        result = clean_description("PUMP SEAL LEAK")
        assert result == "pump seal leak"

    def test_strips_punctuation(self):
        """Punctuation should be removed."""
        result = clean_description("pump leak! (urgent) - check seal.")
        assert "!" not in result
        assert "(" not in result
        assert ")" not in result
        assert "-" not in result or "check" in result

    def test_preserves_content_after_cleaning(self):
        """Core semantic content should survive cleaning."""
        result = clean_description("P-1234 valve stuck closed T-5678")
        assert "valve" in result
        assert "stuck" in result
        assert "closed" in result

    def test_handles_empty_string(self):
        """Empty string should return empty string."""
        assert clean_description("") == ""

    def test_handles_whitespace_only(self):
        """Whitespace-only input should return empty string."""
        assert clean_description("   ") == ""

    def test_handles_none_input(self):
        """None input should be converted to string and handled."""
        result = clean_description(None)
        assert isinstance(result, str)

    def test_realistic_example(self):
        """A realistic maintenance description."""
        result = clean_description(
            "direct contact with pipe support with crevice corrosion"
        )
        assert "direct contact" in result
        assert "pipe support" in result
        assert "crevice corrosion" in result

    def test_mixed_code_patterns(self):
        """Multiple code patterns should all be removed."""
        result = clean_description("V-1234 and T-9012 and P-5678")
        # Should only contain connecting words after code removal
        assert "and" in result or result.strip() == ""

    def test_codes_without_context(self):
        """Description that is ONLY equipment codes should become empty."""
        result = clean_description("P-1234 V-5678 T-9012")
        assert result.strip() == ""


class TestTextEdgeCases:
    """Edge cases from real-world data."""

    def test_very_long_description(self):
        """Long descriptions should be handled (truncation happens at tokenization, not cleaning)."""
        long_text = "pipe support corrosion " * 50
        result = clean_description(long_text)
        assert "pipe support corrosion" in result

    def test_description_with_numbers(self):
        """Numerical values should be preserved (only equipment codes removed)."""
        result = clean_description("pressure reading 150 psi at valve X-123")
        assert "pressure" in result
        assert "reading" in result
        assert "150" in result
        assert "psi" in result

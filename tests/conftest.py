"""
Pytest fixtures and shared test data for pipeline tests.
"""

import os
import sys

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest


# ── Test data ─────────────────────────────────────────────────────────────

SAMPLE_CLEAN_INPUTS = [
    # (input, expected_contains)
    ("Pump P-1234 leak at flange", "leak at flange"),
    ("Valve V-5678 stuck", "valve stuck"),
    ("Tank T-9012 corrosion", "tank corrosion"),
    ("ubolt missing", "ubolt missing"),
    ("thick scaling with severe corrosion", "thick scaling with severe corrosion"),
]

SAMPLE_EMPTY_EDGE_CASES = [
    "",                     # empty string
    "   ",                  # whitespace only
    "P-1234 V-5678",        # codes only (should become empty)
    None,                   # None input
]


@pytest.fixture
def sample_description():
    """A realistic equipment inspection finding."""
    return "direct contact with pipe support with crevice corrosion"


@pytest.fixture
def sample_batch_csv(tmp_path):
    """Create a temporary test batch CSV with known data."""
    import pandas as pd
    data = {
        'DESCRIPTION': [
            'ubolt missing',
            'thick scaling with severe corrosion perforated grating',
            'bolt and nuts depleted',
        ],
        'PRIORITY': ['4', '1', '3'],
        'INITIAL RECOMMENDATION': [
            'to be rectified',
            'to be repaired/ replaced',
            'to be repaired/ replaced',
        ],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "test_batch.csv"
    df.to_csv(path, index=False)
    return path

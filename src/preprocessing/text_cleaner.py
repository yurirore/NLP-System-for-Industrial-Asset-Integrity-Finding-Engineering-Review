"""
Text preprocessing utilities for equipment maintenance descriptions.

Handles equipment code removal, punctuation stripping, and normalization
specific to the PETRONAS equipment maintenance domain.
"""

import re
import string


def clean_description(text: str) -> str:
    """
    Clean equipment description text for topic categorization.

    Removes equipment codes (alphanumeric identifiers like P-1234, V-5678),
    strips punctuation, and lowercases the text.

    Args:
        text: Raw equipment description text.

    Returns:
        Cleaned text with codes removed, punctuation stripped, lowercased.
    """
    text = str(text).lower()

    # Remove equipment codes — regex patterns derived from PETRONAS
    # equipment ID conventions in the training data
    regex_rules = [
        # Equipment codes: single letter, hyphen, 4+ digits  (e.g., P-1234, V-5678)
        r'\b[a-z]-\d{4,}\b',
        # Upstream complex alphanumeric codes
        r'\d{0,3}[a-z]{1,2}\d{1,10}[a-z]{0,3}\d{0,2}[a-z]{0,1}\d[0-2] ',
        r'\d{0,3}[a-z]{0,3}\d{7,10}[a-z]',
        r'\d{1,3}[a-z]{1,3}\d{2,4}',
        r'[a-z]\d{1,4}[a-z]\d{1}',
        r'[a-z]{1,2}\d{4}',
    ]

    for rule in regex_rules:
        text = re.sub(rule, '', text, flags=re.IGNORECASE)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text.strip()

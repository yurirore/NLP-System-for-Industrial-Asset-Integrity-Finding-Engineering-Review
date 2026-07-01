"""
I/O utilities: results export, directory management, error handling.
"""

import os
import pandas as pd


def save_results_to_csv(results: list, output_path: str) -> pd.DataFrame:
    """
    Save pipeline results to CSV with robust error handling.

    Args:
        results: List of result dictionaries from the pipeline.
        output_path: Path where CSV will be saved (relative or absolute).

    Returns:
        DataFrame: The saved results dataframe.
    """
    df_results = pd.DataFrame([
        {
            'DESCRIPTION': r['original_input']['description'],
            'PRIORITY': r['original_input']['priority'],
            'INITIAL_RECOMMENDATION': r['original_input']['initial_recommendation'],
            'PREDICTED_RECOMMENDATION': r['predicted_recommendation'],
            'TOPIC_CATEGORY': r['topic_category'],
            'STRUCTURED_OUTPUT': r['structured_output'],
        }
        for r in results
    ])

    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")

    # Handle locked files (e.g., opened in Excel)
    final_output_path = output_path
    if os.path.exists(output_path):
        print(f"\n⚠️  File already exists: {output_path}")
        print("   Attempting to overwrite...\n")

        try:
            os.remove(output_path)
            print(f"✅ Successfully removed existing file")
        except PermissionError:
            base, ext = os.path.splitext(output_path)
            counter = 1
            while os.path.exists(final_output_path):
                final_output_path = f"{base}_v{counter}{ext}"
                counter += 1
            print(f"❌ Cannot overwrite (locked). Saving as:\n   {final_output_path}")

    # Save to CSV
    df_results.to_csv(final_output_path, index=False)
    absolute_path = os.path.abspath(final_output_path)
    print(f"✅ Results saved: {absolute_path} ({len(df_results)} rows)")
    return df_results

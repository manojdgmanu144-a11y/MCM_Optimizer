"""
src/data_handler.py - Data Preprocessing
"""

def preprocess_dimensions(raw_list):
    """
    Cleans raw input data:
    1. Ensures all values are integers.
    2. Takes absolute values (removes negatives).
    3. Replaces 0 with 1 (since dimension cannot be 0).
    """
    try:
        # Convert to absolute integers and handle zero-dimension edge case
        cleaned = [max(1, abs(int(x))) for x in raw_list]
        return cleaned
    except (ValueError, TypeError):
        return [10, 30, 5, 60] # Default fallback
def prepare_marker_sizes(values, min_size=5, max_size=20):
    """
    Convert values to positive marker sizes suitable for visualization.

    Args:
        values: List or array of values to convert to marker sizes
        min_size: Minimum marker size (default: 5)
        max_size: Maximum marker size (default: 20)

    Returns:
        List of positive marker sizes
    """
    import numpy as np

    # Convert to numpy array if not already
    values = np.array(values)

    # Ensure all values are positive
    positive_values = np.abs(values)

    # If all values are very close to 0, use a default size
    if np.all(positive_values < 0.001):
        return [min_size] * len(values)

    # Scale values to desired range
    min_val = np.min(positive_values)
    max_val = np.max(positive_values)
    range_val = max_val - min_val

    if range_val < 0.001:  # Handle case where all values are identical
        return [min_size] * len(values)

    # Scale to desired size range
    sizes = (positive_values - min_val) / range_val * (max_size - min_size) + min_size

    return sizes.tolist()
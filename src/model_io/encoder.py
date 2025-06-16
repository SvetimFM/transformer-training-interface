def string_to_integer(vocab):
    """
    Converts a vocabulary string to a dictionary mapping each character to its index.

    Args:
        vocab (str): A string containing the vocabulary characters.

    Returns:
        dict: A dictionary mapping each character in the vocabulary to its index.
    """
    return {ch: i for i, ch in enumerate(vocab)}

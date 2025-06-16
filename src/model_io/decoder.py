def integer_to_string(vocab):
    """
    Converts a vocabulary string to a dictionary mapping each index to its character.

    Args:
        vocab (str): A string containing the vocabulary characters.

    Returns:
        dict: A dictionary mapping each index to its character in the vocabulary.
    """
    return {i: ch for i, ch in enumerate(vocab)}

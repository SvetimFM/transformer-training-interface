def prepare_dataset():
    dataset_file = open("./.dependencies/pretraining_dataset.txt", "r")
    dataset = dataset_file.read()

    vocab = sorted(list(set(dataset)))
    charset_size = len(vocab)

    # possible elements of sequences - vocabulary size that the model can learn and then emit
    print("Number of unique characters:", charset_size)
    print(vocab)

    # now need this vocab we will convert to/from this set of characters to a set of integers
    # computers do not understand characters after all :)
    return vocab

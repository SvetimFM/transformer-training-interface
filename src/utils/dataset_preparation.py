import torch


def get_dataset():
    dataset_file = open("./.dependencies/pretraining_dataset.txt", "r")
    return dataset_file.read()


def prepare_vocab():
    dataset = get_dataset()

    vocab = sorted(list(set(dataset)))
    charset_size = len(vocab)

    # possible elements of sequences - vocabulary size that the model can learn and then emit
    print("Number of unique characters:", charset_size)
    print(vocab)

    return vocab

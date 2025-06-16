from utils.dataset_preparation import get_dataset
import torch


def main():
    # Basic encoder - character to integer. Very simple, and lossless encoding method - which does not scale well
    # Would have to encode all relationships between all characters in the dataset
    dataset = get_dataset()
    vocab = sorted(list(set(dataset)))

    # cache the mapping from characters to integers and vice versa (mappings between vocabulary and integers)
    string_to_integer_map = {c: i for i, c in enumerate(vocab)}
    integer_to_string_map = {i: c for i, c in enumerate(vocab)}

    # convert a string to a list of integers and vice versa
    encode = lambda s: [string_to_integer_map[c] for c in s]
    decode = lambda l: "".join([integer_to_string_map[i] for i in l])

    data = torch.tensor(encode(dataset), dtype=torch.long)
    print(data.shape, data.dtype)

    # split the dataset into training and validation sets
    train_size = int(0.85 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    print("Training data size:", train_data.shape)
    print("Validation data size:", val_data.shape)


if __name__ == "__main__":
    main()

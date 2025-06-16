from model_io.decoder import integer_to_string
from model_io.encoder import string_to_integer
from utils.dataset_preparation import prepare_dataset


def main():
    vocab = prepare_dataset()
    print("encoded:", string_to_integer(vocab))
    print("decoded:", integer_to_string(vocab))


if __name__ == "__main__":
    main()

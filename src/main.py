from model_io.decoder import integer_to_string
from model_io.encoder import string_to_integer
from utils.dataset_preparation import prepare_dataset
import tiktoken


def main():
    vocab = prepare_dataset()
    print("encoded:", string_to_integer(vocab))
    print("decoded:", integer_to_string(vocab))

    test_sentence = "Hello, world!"
    print(
        f"Test sentence to integer: {test_sentence} is  {string_to_integer(test_sentence)}\n"
    )

    # lossy, much more efficient encoding method is to have way more tokens for segments of words
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    print(
        f"Test sentence to tiktoken for gpt-3.5-turbo encoding: {test_sentence} is  {encoding.encode(test_sentence)}\n"
    )


if __name__ == "__main__":
    main()

from models.bigram import BigramLM
from utils.training_utils import batchifier
from utils.dataset_preparation import get_dataset
import torch

# hyperparameters

# training
block_size = (
    16  # what is the maximum length of a sequence which influences the next token
)
batch_size = 4  # how many sequences we want to process in parallel
lr = 1e-2  # learning rate
train_split = 0.85  # ratio of training to validation in the training dataset
epochs = 1000
# CUDA support
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Basic encoder - character to integer. Very simple, and lossless encoding method - which does not scale well
    # Would have to encode all relationships between all characters in the dataset - got RAM?
    dataset = get_dataset()
    vocab = sorted(list(set(dataset)))
    vocab_size = len(vocab)

    # cache the mapping from characters to integers and vice versa (mappings between vocabulary and integers)
    string_to_integer_map = {c: i for i, c in enumerate(vocab)}
    integer_to_string_map = {i: c for i, c in enumerate(vocab)}

    # convert a string to a list to numerical mappings  and vice versa
    encode = lambda s: [string_to_integer_map[c] for c in s]
    decode = lambda l: "".join([integer_to_string_map[i] for i in l])

    data = torch.tensor(encode(dataset), dtype=torch.long)
    print(data.shape, data.dtype)

    # split the dataset into training and validation sets
    train_size = int(train_split * len(data))

    train_data = data[:train_size].to(device)
    val_data = data[train_size:].to(device)

    print("Training data size:", train_data.shape)
    print("Validation data size:", val_data.shape)

    # this block contains more than simply 64 characters
    # encoded in the list of integers is the probability distribution of each  character\token appearing given the previous ones
    # this is the crux of attention - context of the next token given relationship with previous x tokens

    # the transformer is limited on prediction of the next token based on the previous tokens in the block window - if it were infinite, that would be nice, but not scalable

    # nice thing too - the chunks impose a limitation that allows parallelization of the training process - as individual chunks contain individual sequences of characters thus different probability distributions

    # (what if we began predicting the next block based on previous blocks?)
    # we would likely gain the same kind of benefit as we would by increasing chunk size in tokenizer

    print(train_data[: block_size + 1])

    # Notice that these are basically the same with an offset of 1
    # this is because all we are doing is simply predicting the next character in the sequence - do this enough times on enough data and generalization across a subset of  problem space in the dataset is approximated

    xb_train, yb_train = batchifier(
        train_data, batch_size=batch_size, block_size=block_size, device=device
    )
    print("Train batch shape:", xb_train.shape, yb_train.shape)
    print("Train batch example:", xb_train[0], yb_train[0])

    xb_val, yb_val = batchifier(
        val_data, batch_size=batch_size, block_size=block_size, device=device
    )
    # print("Validation batch shape:", xb_val.shape, yb_val.shape)
    # print("Validation batch example:", xb_val[0], yb_val[0])

    print("training batch count:", xb_train.shape[0])
    print("validation batch count:", xb_val.shape[0])

    # Want to see the input\output pairs? uncomment the following lines
    # for b in range(batch_size):  # parallelization dimension
    #     for t in range(
    #         block_size
    #     ):  # sequence incrementation dimension (time dimension)
    #         context = xb_train[b, : t + 1]
    #         target = yb_train[b, t]
    #         print(
    #             f"Batch {b}, Time {t}: Context: {context.tolist()}, Target: {target.item()}"
    #         )

    # bigram test
    model = BigramLM(vocab_size, block_size)
    m = model.to(device)
    logits, loss = m(xb_train, yb_train)
    idx = torch.zeros((1, 1), dtype=torch.long)  # start at a space\empty
    idx = idx.to(device)
    # base validation
    print(f"tensor dims: {logits.shape}")
    print(f"loss original: {loss}")
    print(
        f"example output: {decode(m.generate(idx, max_new_tokens=100)[0].tolist())}"
    )  # hack for bigram due to single batch dimension of the indicies

    # lets train! :)
    optimizer = torch.optim.AdamW(
        m.parameters(), lr=lr
    )  # 1e-3 default for this tiny thing, still very fast

    for _ in range(epochs):
        xb, yb = xb_train, yb_train

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # observe improvement in loss and generation!
    print(f"loss post-training: {loss}")
    print(
        f"example output: {decode(m.generate(idx, max_new_tokens=100)[0].tolist())}"
    )  # hack for bigram due to single batch dimension of the indicies

    # observe too the flat peak of the models capability - its time to move on to making use of the rest of the sequences

    # Chapter 2: Self Attention
    # A need for tokens N, N-1,...Ny, is to talk back to tokens N-y
    # simplest way is to average\sum all tokens before N token
    # simple == lossy, but its a start :)

    # a lame way of doing things is the 'modern' way - loop
    # xbow = torch.zeros((B,T,C))
    # for b in range(B):
    #     for t in range(T):
    #         xprev = x[b,:t+1]
    #         xbow[b,t] = torch.mean(xprev,0)
    # This gives us a mean aggregate of the previous tokens for inference of next token
    # but this does not parallelize nicely, not in a for loop - unless you want to build an optimization no one has bothered to make because seven people need this actually
    # So we turn to linear algebra

    # B, T, C = 4,8,2 # example only
    # x = torch.randn(B,T,C)
    # weights = torch.tril(torch.ones(T,T)) # create the matrix that allows sequence based contributions to the next token
    # weights = weights / weights.sum(1, keepdim=True) # equally average all previous terms - basic
    # xbow = weights @ x # dot multiply - now its parallel out of the box! :)

    # third version!
    # wei = torch.zeros((T, T))
    # wei = wei.masked_fill(tril == 0, float("inf"))
    # wei = F.softmax(wei, dim=-1) # this normalizes -inf to 0 and 0's to 1
    # xbow = wei @ x
    # tril = torch.tril()

    # Note - all these still equally consider each token in a given sequence
    # additionally, the token weights will get modified as they are now related to each other
    # this is affinity - and this is the relationship of tokens to each other given sequence\context

    # matrix multiplication of the weights in a triangular fashion provides a map of how to consider tokens with each other


if __name__ == "__main__":
    main()

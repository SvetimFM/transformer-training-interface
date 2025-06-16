import torch


# returns a batch of data for training or validation
# in comes a tensor out comes tensor in specified chunks
def batchifier(data, batch_size, block_size):
    torch.manual_seed(1337)

    # data is a tensor of integers representing characters
    # batch_size is the number of sequences in a batch (for parallelization)
    # block_size is the number of characters in each sequence
    sample_points = torch.randint(len(data) - block_size, (batch_size,))

    inputs = torch.stack([data[i : i + block_size] for i in sample_points])
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in sample_points])

    return inputs, targets

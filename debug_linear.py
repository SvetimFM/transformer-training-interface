import torch
import torch.nn.functional as F

# Shapes
batch_size = 2
seq_len = 10
n_embed = 128
hidden_dim = 512

# Create tensors
z_hidden = torch.randn(batch_size, seq_len, hidden_dim)
W_1 = torch.randn(n_embed, hidden_dim)
b_1 = torch.zeros(n_embed)

print(f"z_hidden shape: {z_hidden.shape}")
print(f"W_1 shape: {W_1.shape}")
print(f"W_1.t() shape: {W_1.t().shape}")
print(f"b_1 shape: {b_1.shape}")

# F.linear(input, weight, bias) expects:
# input: (*, in_features)
# weight: (out_features, in_features)
# So for z_hidden @ W -> output:
# z_hidden: (batch, seq, hidden_dim)
# W should be: (n_embed, hidden_dim) to get (batch, seq, n_embed)

try:
    # This should work
    result = F.linear(z_hidden, W_1, b_1)
    print(f"\nF.linear(z_hidden, W_1, b_1) shape: {result.shape}")
except Exception as e:
    print(f"Error with W_1: {e}")

try:
    # This is what I was doing
    result = F.linear(z_hidden, W_1.t(), b_1)
    print(f"\nF.linear(z_hidden, W_1.t(), b_1) shape: {result.shape}")
except Exception as e:
    print(f"Error with W_1.t(): {e}")
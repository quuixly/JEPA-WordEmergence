import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, dimension):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension
        # During learning only selected rows will be updated (see forward method)
        self.weights = nn.Parameter(torch.randn(vocabulary_size, dimension), requires_grad=True)

    def forward(self, x):
        # Return vectors for each word in x
        # x.shape = (batch_size, input_sequence_length)
        # return shape = (batch_size, input_sequence_length, dimension)
        return self.weights[x]


class PositionalEncoding(nn.Module):
    def __init__(self, max_sequence_length, dimension):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.dimension = dimension
        # In the original "Attention Is All You Need" paper, sinusoidal positional encodings were used.
        # Here, we use learnable positional embeddings.
        self.weights = nn.Parameter(torch.zeros(max_sequence_length, dimension), requires_grad=True)

    def forward(self, x):
        # x.shape (bath_size, input_sequence_length, dimension)
        # return shape = (batch_size, input_sequence_length, dimension)
        if x.dim() == 2: # Without batches
            input_sequence_length = x.shape[0]
            positional_vectors = self.weights[:input_sequence_length, :]

            return x + positional_vectors
        elif x.dim() == 3:
            batch_size, input_sequence_length, dimension = x.shape

            positional_vectors = self.weights[:input_sequence_length, :]
            # Explicit broadcasting (input_sequence_length, dimension) to align the batch_size
            positional_vectors = positional_vectors.unsqueeze(0).expand(batch_size, input_sequence_length, dimension)

            return x + positional_vectors
        else:
            raise ValueError("x must be 2D or 3D tensor")

class GPT(nn.Module):
    def __init__(self, dimension=512, num_heads=8, num_layers=4, vocabulary_size=61, max_sequence_length=61):
        super().__init__()

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
        # Return vectors for each word x_i in x
        # x.shape = (batch_size, x_n)
        # return shape = (batch_size, x_n, dimension)
        return self.weights[x]


class GPT(nn.Module):
    def __init__(self, dimension=512, num_heads=8, num_layers=4, vocabulary_size=61, sequence_length=61):
        super().__init__()


if __name__ == "__main__":
    pass
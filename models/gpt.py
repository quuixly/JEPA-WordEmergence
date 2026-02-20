import torch
import torch.nn as nn
import numpy as np


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, dimension):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension
        # During learning only selected rows will be updated (see forward method)
        self.weights = nn.Parameter(torch.randn(vocabulary_size, dimension), requires_grad=True)

    def forward(self, x):
        # Return vectors for each word in x
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
        batch_size, input_sequence_length, dimension = x.shape

        positional_vectors = self.weights[:input_sequence_length, :]

        return x + positional_vectors


class Linear(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.weights = nn.Parameter(torch.randn(output_dimension, input_dimension), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(output_dimension), requires_grad=True)

    def forward(self, x):
        # Why transposition? Well, self.weights could be (input_dimension, output_dimension), but we do in this way
        # for performance reasons
        # https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277/3
        return x @ self.weights.T + self.bias


class Softmax(nn.Module):
    # Converts the input into a probability distribution along the specified dimension
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Subtract max for numerical stability to avoid large exponentials
        x_max = torch.max(x, dim=self.dim, keepdim=True)[0]
        x_exp = torch.exp(x - x_max)

        # Divide by the sum along the dimension to get probabilities
        return x_exp / torch.sum(x_exp, dim=self.dim, keepdim=True)


class MultiHeadAttention(nn.Module):
    # We train the model to transform the input X into Query, Key, and Value matrices:
    # Query - each row represents the "question" that the corresponding row in X wants to ask
    # Key - each row represents information about the corresponding row in X ("description of itself")
    # Value - each row represents the contextual information of the corresponding row in X
    # (in the context of this specific input X)
    #
    # Then we multiply the Query matrix by the Key matrix
    # (see https://www.youtube.com/watch?v=XkY2DOUCWMU for an intuitive explanation, why we use matrix multiplication).
    # Think of matrix multiplication as a transformation: each question from Query is "asked" to every description in Key.
    #
    # Next, we apply softmax to generate a probability distribution.
    # This tells us which contextual information from Value is important and with what weight.
    #
    # Finally, we multiply this distribution by Value to produce the attention output.
    #
    # This is just the intuition behind the attention mechanism.
    # Note that it could be computed differently, and the model itself does not "know" that these matrices
    # are Query, Key, or Value—it may learn something else entirely.
    def __init__(self, dimension, num_heads = 2, masked_attention=False):
        super().__init__()
        self.dimension = dimension
        # If True, the attention is masked (Masked Multi-Head Attention)
        self.masked_attention = masked_attention
        # If num_heads == 1, this is just a standard self-attention mechanism
        # (without projections and combining heads, but we assume that we will use always num_heads > 1)
        # If num_heads > 1, this becomes multi-head attention (split Q/K/V into multiple heads)
        self.num_heads = num_heads

        # Linear projections for query, key, and value
        self.weights_query = Linear(dimension, dimension)
        self.weights_key = Linear(dimension, dimension)
        self.weights_value = Linear(dimension, dimension)
        # Final output projection after combining heads
        self.weights_output_projection = Linear(dimension, dimension)
        # Softmax to convert attention scores into probabilities
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        # Compute Q, K, V matrices
        Q = self.weights_query(x)
        K = self.weights_key(x)
        V = self.weights_value(x)

        # Split each into multiple heads
        batch_size, input_sequence_length, dimension = x.shape
        head_dimension = dimension // self.num_heads
        # Reshape and transpose to (batch_size, num_heads, input_sequence_length, head_dimension)
        # This avoids creating many smaller matrices manually
        # subsequent operations can be performed on each head in parallel
        Q = Q.view(batch_size, input_sequence_length, self.num_heads, head_dimension).transpose(1, 2)
        K = K.view(batch_size, input_sequence_length, self.num_heads, head_dimension).transpose(1, 2)
        V = V.view(batch_size, input_sequence_length, self.num_heads, head_dimension).transpose(1, 2)

        # Scaled dot-product attention for each head
        # Scale by sqrt(head_dimension) to avoid extreme values,
        # which helps prevent vanishing or exploding gradients
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(head_dimension)

        # Masked attention
        if self.masked_attention:
            mask = torch.tril(torch.ones(input_sequence_length, input_sequence_length, device=x.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Convert scores to probabilities and apply to values
        attention = self.softmax(scores) @ V

        # Merge multiple heads back into a single tensor
        attention = attention.transpose(1, 2).contiguous().view(batch_size, input_sequence_length, dimension)

        # Final linear projection
        attention = self.weights_output_projection(attention)

        return attention


class LayerNorm(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        if isinstance(dimension, int):
            dimension = (dimension,)
        self.dimension = dimension
        self.eps = 1e-6

        # Learnable scaling (gamma) and shifting (beta) parameters
        # They allow the model to adjust the normalized output
        self.gamma = nn.Parameter(torch.ones(*dimension), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(*dimension), requires_grad=True)

    def forward(self, x):
        # Standardize x to mean 0 and standard deviation 1 along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + self.eps)
        x_normalized = (x - mean) / std

        # Allow the model to learn an optimal scale and shift after normalization
        out = self.gamma * x_normalized + self.beta

        return out


class AddAndNorm(nn.Module):
    # Residual connection followed by Layer Normalization
    def __init__(self, dimension):
        super().__init__()
        self.layer_norm = LayerNorm(dimension)

    def forward(self, x, sublayer_output):
        # Add residual connection (skip connection)
        # Then apply LayerNorm to stabilize training
        return self.layer_norm(x + sublayer_output)


class GELU(nn.Module):
    # Almost the same as ReLU, but with a smooth transition around x = 0,
    # which promotes more stable gradient propagation
    # and often better convergence in large models.
    # https://arxiv.org/abs/1606.08415
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)))



class Dropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        # Dropout should only be applied during training
        # (we don't want to randomly deactivate neurons during inference)
        if self.training:
            # For each neuron, generate a random number in [0, 1]
            # and zero it out with probability p
            mask = (torch.rand_like(x) > self.p).float()
            # Scale the remaining activations by 1/(1-p)
            # to preserve the expected value of activations
            x = x * mask / (1.0 - self.p)

        return x


class FeedForward(nn.Module):
    # Simple two-layer feedforward network, used in Transformers
    def __init__(self, dimension, dropout = 0.01):
        super().__init__()
        hidden_dimension = dimension * 4 # Typical expansion factor in Transformer FFN

        self.linear1 = Linear(dimension, hidden_dimension)
        self.linear2 = Linear(hidden_dimension, dimension)
        self.dropout = Dropout(dropout)
        self.activation_function = GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_function(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class Decoder(nn.Module):
    def __init__(self, dimension, num_heads):
        super().__init__()

        self.masked_multi_head_attention = MultiHeadAttention(dimension, num_heads, True)
        self.add_and_norm1 = AddAndNorm(dimension)
        self.feed_forward = FeedForward(dimension)
        self.add_and_norm2 = AddAndNorm(dimension)

    def forward(self, x):
        masked_attention_output = self.masked_multi_head_attention(x)
        x = self.add_and_norm1(x, masked_attention_output)
        feed_forward_output = self.feed_forward(x)
        x = self.add_and_norm2(x, feed_forward_output)

        return x


class GPT(nn.Module):
    def __init__(self, dimension=512, num_heads=8, num_layers=4, vocabulary_size=61, max_sequence_length=61):
        super().__init__()

        self.embedding = Embedding(vocabulary_size, dimension)
        self.positional_encoding = PositionalEncoding(max_sequence_length, dimension)
        self.blocks = nn.ModuleList([Decoder(dimension, num_heads) for i in range(num_layers)])
        self.head = Linear(dimension, vocabulary_size)

    def forward(self, x):
        embedding = self.embedding(x)
        x = self.positional_encoding(embedding)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)

        return x
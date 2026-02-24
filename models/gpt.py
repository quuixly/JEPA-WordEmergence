import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.dataset import OthelloDataset
from models.trainer import Trainer


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(attn_output)


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, dropout_rate=0.1):
        super().__init__()
        self.attention_layer = MaskedMultiHeadAttention(d_model, n_heads, dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout_rate)
        )
        self.add_and_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attention_layer(self.norm(x))
        x = x + self.feed_forward(self.add_and_norm(x))

        return x


class GPT(nn.Module):
    def __init__(self, n_layers = 4, n_heads = 8, d_model = 512, vocabulary_size = 63, context_window=63, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(context_window, d_model))
        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(n_heads, d_model, dropout_rate) for _ in range(n_layers)]
        )
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocabulary_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        batch_size, sequence_length = x.shape

        x = self.embedding(x)
        x = x + self.positional_encoding[:sequence_length, :]
        for layer in self.decoder_stack:
            x = layer(x)
        x = self.head_norm(x)
        y = self.head(x)

        return y

    def get_optimizer(self, weight_decay=0.005, lr=3e-4, betas=(0.9, 0.95)):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        no_decay.add('positional_encoding')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)

    def get_loss_fn(self):
        return nn.CrossEntropyLoss(ignore_index=61)


if __name__ == "__main__":
    dataset = OthelloDataset()
    model = GPT()
    trainer = Trainer(dataset, model, 32, 1, True)
    trainer.train()
# Causal multi-head self-attention for one GPT transformer block.
from torch import nn
from torch.nn import functional as F

import gpt_config as config


# Project queries, keys and values into heads and attend only to earlier positions.
class CausalSelfAttention(nn.Module):
    # Create the attention projections and dropout layers.
    def __init__(self, n_head=config.N_HEAD, d_model=config.D_MODEL, dropout=config.DROPOUT):
        super().__init__()
        self.n_head = n_head
        self.head_size = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    # Return causal self-attention values with the original input shape.
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.n_head, self.head_size)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        dropout = self.attn_dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.resid_dropout(self.proj(y))
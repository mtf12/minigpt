# Pre-norm transformer block with a four-times-width GELU feed-forward network.
from torch import nn

import gpt_config as config
import attention


# Normalize before attention and the MLP, then add each result back residually.
class TransformerBlock(nn.Module):
    # Create the layer norms, attention and feed-forward network.
    def __init__(self, n_head=config.N_HEAD, d_model=config.D_MODEL, dropout=config.DROPOUT):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = attention.CausalSelfAttention(n_head, d_model, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    # Apply the two pre-norm residual sublayers.
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
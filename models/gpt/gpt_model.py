# Decoder-only GPT with learned positions, pre-norm blocks and tied input/output embeddings.
import math

import torch
from torch import nn
from torch.nn import functional as F

import gpt_config as config
import transformer_block


# Pick CUDA (Colab) first, then Apple MPS, then CPU. Used by run_gpt and the training.
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Build a GPT model with configurable dimensions for small CPU tests.
class GPTModel(nn.Module):
    # Create embeddings, transformer blocks, final norm and the tied output layer.
    def __init__(self, vocab_size=config.VOCAB_SIZE, pad_id=config.PAD_ID, eos_id=config.EOS_ID,
                 n_layer=config.N_LAYER, n_head=config.N_HEAD, d_model=config.D_MODEL,
                 context_length=config.CONTEXT_LENGTH, dropout=config.DROPOUT):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([transformer_block.TransformerBlock(n_head, d_model, dropout) for _ in range(n_layer)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

        residual_std = 0.02 / math.sqrt(2 * n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp[2].weight, mean=0.0, std=residual_std)

    # Initialize linear and embedding weights with the GPT normal distribution.
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module is not self.lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # Return next-token logits and optional cross-entropy for already-shifted targets.
    def forward(self, ids, targets=None):
        _, seq_len = ids.shape
        positions = torch.arange(seq_len, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=self.pad_id)
        return logits, loss

    # Count all parameters once and return the total and non-embedding counts.
    def count_parameters(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        embeddings = self.token_embedding.weight.numel() + self.position_embedding.weight.numel()
        return total, total - embeddings

    # Sample next tokens with temperature and top-k, stopping when EOS is generated.
    def generate(self, ids, max_new_tokens=config.MAX_NEW_TOKENS, temperature=config.TEMPERATURE, top_k=config.TOP_K):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self(ids[:, -self.context_length:])
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, -1:]] = float("-inf")

                probabilities = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
                ids = torch.cat((ids, next_id), dim=1)
                if torch.any(next_id == self.eos_id):
                    break
        return ids

# GPT training settings. Token caches and checkpoints can be moved with environment variables (Colab does this).
from pathlib import Path
import os
import sys

# Paths (this file lives in models/training/)
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parents[1]

# The training reuses the GPT model, the tokenizer and the preprocessing helpers.
for folder in ["gpt", "tokenizer", "preprocessing"]:
    if str(PROJECT_ROOT / "models" / folder) not in sys.path:
        sys.path.append(str(PROJECT_ROOT / "models" / folder))

import gpt_config
import tokenizer_config

TRAIN_PATH = tokenizer_config.TRAIN_PATH
VAL_PATH = tokenizer_config.VAL_PATH
TOKENIZER_PATH = tokenizer_config.TOKENIZER_PATH
ARTIFACTS_DIR = Path(os.getenv("MINIGPT_ARTIFACTS_DIR") or TRAINING_DIR / "artifacts")  # token caches
CHECKPOINTS_DIR = Path(os.getenv("MINIGPT_CHECKPOINTS_DIR") or ARTIFACTS_DIR)  # on Colab: Google Drive
TRAIN_TOKENS_PATH = ARTIFACTS_DIR / "train_tokens.u16"
VAL_TOKENS_PATH = ARTIFACTS_DIR / "val_tokens.u16"
LAST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "last_checkpoint.pt"  # where training resumes from
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pt"  # the epoch with the lowest validation loss
HISTORY_PATH = CHECKPOINTS_DIR / "history.json"  # train / validation loss of the whole run, kept across resumes
SAVED_TOKENIZER_PATH = CHECKPOINTS_DIR / "tokenizer.json"  # the tokenizer the checkpoints were trained with

# Data
CONTEXT_LENGTH = gpt_config.CONTEXT_LENGTH  # every training window fills the whole model context
PAD_ID = gpt_config.PAD_ID
CACHE_BUFFER_TOKENS = 100_000  # tokens kept in memory before they are written to the cache (~200 KB)

# Training
EPOCHS = 3  # 3 x 71.6M train tokens = ~215M tokens, ~9 tokens per parameter of the 23M-parameter model
BATCH_SIZE = 512  # windows per update = ~131k tokens, 547 updates per epoch
MICRO_BATCH_SIZE = 16  # windows per forward pass (4.4 GB measured on the 8 GB M1); 32 of them make one update
LEARNING_RATE = 3e-4  # peak learning rate, a common value for GPT models of this size
WARMUP_STEPS = 100  # ~6% of the 1,641 updates: small steps first while the weights are still random
MIN_LEARNING_RATE = 3e-5  # the cosine ends at 10% of the peak instead of 0, so the last updates still learn
WEIGHT_DECAY = 0.1  # AdamW pulls the weights a little towards 0 to limit overfitting
BETA1 = 0.9  # how much of the past gradients the momentum keeps
BETA2 = 0.95  # lower than the default 0.999 so Adam reacts faster to loss spikes
GRAD_CLIP = 1.0  # gradients longer than this are shrunk, so one bad batch cannot blow up the weights
SEED = 42
CHECKPOINT_MINUTES = 15  # a dropped Colab session loses at most this much training

# After training
SAMPLE_TOKENS = 60  # tokens generated per prompt: about one or two sentences

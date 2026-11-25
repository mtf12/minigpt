# GPT model settings. Vocabulary and special-token ids come from the trained tokenizer.
from pathlib import Path
import sys

from tokenizers import Tokenizer

# Paths
GPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GPT_DIR.parents[1]
TOKENIZER_DIR = PROJECT_ROOT / "models" / "tokenizer"
PREPROCESSING_DIR = PROJECT_ROOT / "models" / "preprocessing"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"
TOKENIZER_PATH = TOKENIZER_DIR / "artifacts" / "tokenizer.json"
VAL_PATH = SPLITS_DIR / "val.parquet"

for path in (TOKENIZER_DIR, PREPROCESSING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tokenizer_config

_tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB_SIZE = _tokenizer.get_vocab_size()
PAD_ID = _tokenizer.token_to_id(tokenizer_config.PAD_TOKEN)
BOS_ID = _tokenizer.token_to_id(tokenizer_config.BOS_TOKEN)
EOS_ID = _tokenizer.token_to_id(tokenizer_config.EOS_TOKEN)
del _tokenizer

# Model size
N_LAYER = 6  # Six blocks keep the model small enough for an 8 GB machine.
N_HEAD = 6  # Six heads divide the 384-wide representation into 64 dimensions each.
D_MODEL = 384
CONTEXT_LENGTH = 256  # A short context keeps attention memory manageable on 8 GB.
DROPOUT = 0.1

# Generation
SEED = 42
DEMO_BATCH_SIZE = 2  # One Arabic and one English validation article.
MAX_NEW_TOKENS = 24  # Short samples keep the untrained model demo quick.
TEMPERATURE = 0.8
TOP_K = 40
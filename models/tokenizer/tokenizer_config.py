# Settings of the tokenizer stage. Named tokenizer_config (not config) so it never clashes with preprocessing/config.py.
from pathlib import Path
import sys

# Paths (this file lives in models/tokenizer/)
TOKENIZER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOKENIZER_DIR.parents[1]
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"
TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH = SPLITS_DIR / "val.parquet"
ARTIFACTS_DIR = TOKENIZER_DIR / "artifacts"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"
VOCAB_BENCHMARK_PATH = ARTIFACTS_DIR / "vocab_benchmark.json"

# The tokenizer reuses parquet_io (batch reading) and light_cleaning (normalize) from preprocessing.
PREPROCESSING_DIR = PROJECT_ROOT / "models" / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.append(str(PREPROCESSING_DIR))

LANGUAGES = ["ar", "en"]

# BPE
VOCAB_SIZE = 32_000
VOCAB_SIZE_CANDIDATES = [16_000, 24_000, 32_000, 48_000]  # compared by run_benchmark
MIN_FREQUENCY = 2  # a pair must appear at least twice to be merged

# Special tokens (their ids are 0, 1, 2, 3 in this order)
UNK_TOKEN = "<|unk|>"
BOS_TOKEN = "<|bos|>"  # added at the start of every article
EOS_TOKEN = "<|eos|>"  # added at the end of every article
PAD_TOKEN = "<|pad|>"  # only for batching
SPECIAL_TOKENS = [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]

from pathlib import Path

# Paths (this file lives in models/preprocessing/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"          # original data, never modified
CLEANED_DIR = DATASETS_DIR / "cleaned"  # cleaned + balanced ar/en
SPLITS_DIR = DATASETS_DIR / "splits"    # train / val / test

LANGUAGES = ["ar", "en"]
SEED = 42
BATCH_SIZE = 2000  # articles in memory at a time, in every step

# Cleaning
MIN_CHARS = 150  # articles shorter than this are dropped
WORKERS = 2      # processes used for cleaning

# Balancing: tokens are estimated from characters until we have a tokenizer
CHARS_PER_TOKEN = {"ar": 3.3, "en": 4.0}
BALANCE_TOLERANCE = 0.02  # a language can be 2% bigger than the other before we trim it

# Splits
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1  # test gets the rest

# miniGPT — Data Preparation, Tokenizer, GPT Model & Training

> **🚧 Work in progress.** Data preparation, the tokenizer, the GPT model and its training pipeline are in place.
> Full training and final test evaluation have not been run yet.
>
> - [x] Data preparation: cleaning, language balancing, train / val / test splits
> - [x] Tokenizer (trained on Arabic + English together)
> - [x] Model (GPT, from scratch)
> - [x] Training pipeline (train split, validation loss and checkpoints)
> - [ ] Full training and final test evaluation (per language)

A bilingual (Arabic + English) GPT built from scratch. The project prepares 277,739 Wikipedia articles into a
~93M-token corpus, trains one shared 32k BPE tokenizer, and provides a training pipeline for the GPT model.

## Key findings & decisions

- **Found a cleaning rule that silently deleted most of the text in ~18% of Arabic articles** (measured on a 20k-article sample).
  Headings like "أحداث" (Events) were treated as the end of the article, so everything after them was cut.
  Fixed it so only real end sections (references, external links) are cut, recovered **1,282 Arabic articles**,
  and added regression tests so it cannot come back.
- **Balanced the two languages by tokens, not by article count.** Arabic articles are ~5x longer than English ones
  (5,048 vs 1,005 characters on average). An equal number of articles would have made Arabic **~86% of the training tokens**.
  Balancing by tokens gives a ~50/50 mix (47.0M vs 46.3M estimated tokens) without throwing away any data.
- **Light normalization that keeps real Arabic.** Tashkeel (found in 68% of Arabic articles), `ة` vs `ه` and hamza forms
  are kept so the model learns them. Only invisible characters and stretching tatweel (`محـــمد`) are removed,
  while the meaningful tatweel in the Hijri date marker `هـ` (~42,000 uses in the raw data) is kept.
- **Fixed number merging that changed meaning.** `5 434 738` became `5434 738`, and table cells on different lines were
  glued into one number. Now whole numbers are joined, and only on the same line.
- **Built to scale.** Every stage streams the data in batches of 2,000 articles (peak memory ~780 MB for the full corpus),
  and every refactor was checked to produce row-for-row identical output.
- **Picked the vocabulary size by measuring it.** Trained the tokenizer with 16k, 24k, 32k and 48k tokens and counted
  how many tokens each one needs for the validation split. Going from 16k to 32k cut the Arabic tokens by 10.6% and the
  English ones by 8.3%. After that the gain slows down: 48k saves only another 5.3% (Arabic) and 4.0% (English),
  but makes the embedding and output layers 50% bigger, which is a bad deal for a small model. So the vocabulary is 32,000.
- **One vocabulary that is fair to both languages.** Of the 32,000 tokens, 15,867 are Arabic and 15,023 English
  (the rest are digits, punctuation and byte pieces). An Arabic token covers 3.8 characters on average and an English
  one 4.0, so Arabic costs only ~6% more tokens for the same amount of text. Common words like ` عاصمة` and ` العربية`
  are a single token.
- **Byte-level, so no character is ever unknown.** Text is split into UTF-8 bytes before BPE, so tashkeel, rare letters
  or anything else is encoded and decoded back exactly. New text (a prompt) goes through the same `normalize()`
  as the training data first, so the model never gets forms it did not see in training (`محـــمد` instead of `محمد`).
- **Used pre-norm transformer blocks.** The 6-layer, 6-head model has a width of 384, a 256-token context and 0.1 dropout.
  Each block normalizes before attention and the MLP, while both residual paths remain direct.
- **Sized the model for an 8 GB laptop and a free Colab GPU.** 6 layers, 6 heads, width 384 and a 256-token context
  give 23,033,856 parameters. One forward and backward pass on 16 windows uses 4.4 GB on an 8 GB M1, and 3 epochs over
  the 71.6M train tokens show the model ~215M tokens (~9 per parameter).
- **Tied the input and output token weights.** The 32,000 × 384 token matrix is shared, saving 12,288,000 parameters.
  The measured model has 23,033,856 parameters total: 10,647,552 outside embeddings and 12,386,304 in token and position embeddings.
- **Trained by epochs, on the train split only.** Each epoch shuffles the 279,757 windows of 256 tokens and visits
  each one once, then measures loss, accuracy and perplexity on validation. One update is 512 windows (~131k tokens),
  added up from 32 micro-batches of 16, so an epoch is 547 updates. AdamW warms up over 100 updates to 3e-4, then
  goes down to 3e-5 with a cosine. The test split is never read during training.
- **Training can stop and resume, on Colab too.** The last checkpoint (weights, optimizer and the position inside the
  epoch) is saved every 15 minutes and after every epoch, through a temporary file so a dropped session never leaves
  a broken one. On Colab the checkpoints live on Google Drive, so a new session continues at the exact next update.
  A test checks that a resumed run ends with the same weights as a run that never stopped. The checkpoint also keeps
  the loss of every update in `history.json`, so the loss curve covers the whole run across sessions. The tokenizer
  of the run is saved next to the checkpoints, and a checkpoint refuses to be used with any other `tokenizer.json`. On CUDA the model trains in float16
  mixed precision.

## Results

| | Arabic | English |
|---|---|---|
| Raw articles | 35,952 | 241,787 |
| After cleaning | 30,733 (85.5%) | 184,332 (76.2%) |
| Estimated tokens | ~47.0M | ~46.3M |
| Train / val / test | 24,586 / 3,073 / 3,074 | 147,465 / 18,433 / 18,434 |

No article appears in two splits, and each split keeps the same Arabic / English ratio.

GPT model (before training):

| Configuration | Total parameters | Non-embedding | Embeddings |
|---|---:|---:|---:|
| 6 layers, 6 heads, width 384, context 256 | 23,033,856 | 10,647,552 | 12,386,304 |

An untrained forward pass on a two-article validation batch ran on MPS with loss 10.4609.

Tokenizer (32,000 tokens), measured on the validation split:

| | Arabic | English |
|---|---|---|
| Tokens in the vocabulary | 15,867 | 15,023 |
| Characters per token | 3.76 | 3.98 |
| Validation tokens | 4,250,589 | 4,713,475 |

| Vocabulary | 16k | 24k | 32k | 48k |
|---|---|---|---|---|
| Arabic validation tokens | 4,753,922 | 4,441,681 | **4,250,589** | 4,027,431 |
| English validation tokens | 5,142,548 | 4,871,789 | **4,713,475** | 4,523,325 |

## Data source

Both languages come from the Hugging Face dataset [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), dump of November 2023:

| Language | Config | Articles used |
|---|---|---|
| Arabic | `20231101.ar` (Arabic Wikipedia) | The first 35,952 of 1,219,201 articles, in dataset order |
| English | `20231101.simple` (Simple English Wikipedia) | All 241,787 articles |

Only the `id`, `title` and `text` columns are kept. The text is licensed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GFDL](https://www.gnu.org/licenses/fdl-1.3.html).

Known limitations:
- The Arabic part is the first rows of the dataset, not a random sample, so it may not represent all of Arabic Wikipedia.
  About 97% of the Arabic articles are still unused.
- The two sources differ in style: Simple English Wikipedia uses short, easy sentences, while Arabic Wikipedia is written
  for a general adult audience. This is one reason Arabic articles are ~5x longer.

## How to run

```bash
pip install -r requirements.txt

# data preparation: everything, with the reports
python models/preprocessing/run_preprocessing.py

# data preparation: cleaning, balancing and splitting only
python models/preprocessing/run_preprocessing.py --skip-reports

# tokenizer: train, then measure on the validation split (needs datasets/splits/)
python models/tokenizer/run_tokenizer.py

# tokenizer: also compare 16k / 24k / 32k / 48k vocabularies (slow)
python models/tokenizer/run_tokenizer.py --benchmark

# GPT model: run a validation forward pass and short Arabic / English generations
python models/gpt/run_gpt.py

# training: cache train / validation tokens, then train and save checkpoints
python models/training/run_training.py

# run one epoch as a shorter check
python models/training/run_training.py --epochs 1

# resume training from the latest checkpoint
python models/training/run_training.py --resume
```

Or open the [notebook](models/data_pipeline.ipynb) and run the cells in order, locally or on Google Colab.

The raw data lives in `datasets/raw/ar.parquet` and `datasets/raw/en.parquet` and is never modified.
Set `MINIGPT_ARTIFACTS_DIR` to move token caches, or `MINIGPT_CHECKPOINTS_DIR` to move checkpoints independently.

### On Google Colab

Upload these files once to Google Drive, keeping the folder names:

| Drive | From the project | Size | Used by |
|---|---|---|---|
| `My Drive/miniGPT/raw/ar.parquet`, `en.parquet` | `datasets/raw/` | 97 + 107 MB | notebook stages 1-6 |
| `My Drive/miniGPT/splits/train.parquet`, `val.parquet` | `datasets/splits/` | 148 + 19 MB | training without re-running stages 1-6 |

Then open the notebook on Colab (File > Open notebook > GitHub, `mtf12/minigpt`, `models/data_pipeline.ipynb`),
choose a GPU runtime and run all cells. The Setup cell clones the repository, installs the requirements, mounts Drive
and copies the data. Checkpoints, `history.json` and the `tokenizer.json` they were trained with are saved to
`My Drive/miniGPT/checkpoints/`.
If the session drops, run the Setup cell and then the training cell again: training continues from the last checkpoint.

To use the trained model locally, download `best_model.pt` and `tokenizer.json` from `My Drive/miniGPT/checkpoints/`
into `models/training/artifacts/`, then run cell 13 (`training_report.generate_samples`). It always reads the
tokenizer saved next to the model, not the one in `models/tokenizer/artifacts/`.

To only train, without the notebook, run in a Colab cell:

```
!GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/mtf12/minigpt.git
%cd minigpt
!pip install -q -r requirements.txt
!python models/training/colab_runner.py
```

## Stages

| # | Stage | File | What it does |
|---|---|---|---|
| 1 | Exploration | [data_exploration.py](models/preprocessing/data_exploration.py) | Size, lengths, leftover problems, a sample article, token estimate (read only) |
| 2 | Light cleaning | [light_cleaning.py](models/preprocessing/light_cleaning.py) | Normalization (NFC, invisible characters, stretching tatweel) and basic cleaning (wiki markup, citations, reference sections). Never removes whole sections |
| 3 | Heavy cleaning | [heavy_cleaning.py](models/preprocessing/heavy_cleaning.py) | Structural cleaning: tables, lists, date lists (events / births / deaths), empty headings |
| 2–3 | Run the cleaning | [clean_dataset.py](models/preprocessing/clean_dataset.py) | Runs light then heavy cleaning on the whole dataset; drops articles under 150 characters after each step and exact duplicates; writes `datasets/cleaned/` |
| 4 | Quality report | [quality_report.py](models/preprocessing/quality_report.py) | Raw vs. cleaned: what was removed and why, leftover problems, before / after examples |
| 5 | Balancing | [balance_languages.py](models/preprocessing/balance_languages.py) | If one language has more than 2% more tokens, random articles are removed from it |
| 6 | Splitting | [split_data.py](models/preprocessing/split_data.py) | 80 / 10 / 10 per language, Arabic and English mixed in every split, written to `datasets/splits/` |
| 7 | Tokenizer training | [train_tokenizer.py](models/tokenizer/train_tokenizer.py) | Byte-level BPE with 32,000 tokens, trained on the train split with every article as `title\n\ntext`. Saves `models/tokenizer/artifacts/tokenizer.json` |
| 8 | Tokenizer evaluation | [evaluate_tokenizer.py](models/tokenizer/evaluate_tokenizer.py) | Tokens per character for each language on the validation split; optional comparison of vocabulary sizes saved to `artifacts/vocab_benchmark.json` |
| 9 | Encode / decode | [encode_text.py](models/tokenizer/encode_text.py) | Encodes new text after the same `normalize()` as the training data and decodes it back; shows a live example |
| 10 | GPT model | [gpt_model.py](models/gpt/gpt_model.py) | Decoder-only pre-norm GPT with causal attention, tied token embeddings and temperature / top-k generation |
| 11 | GPT training | [train_model.py](models/training/train_model.py) | Caches token ids, shuffles and visits every train window once per epoch, AdamW with warmup and cosine learning rate, validation after every epoch, checkpoints and `history.json` every 15 minutes and after every epoch |
| 12 | Loss curves | [training_report.py](models/training/training_report.py) | Train loss of every update and validation loss of every epoch, drawn from `history.json` |
| 13 | Samples | [training_report.py](models/training/training_report.py) | Loads the best model (lowest validation loss) and continues Arabic and English prompts |

Shared files: [config.py](models/preprocessing/config.py) (paths and settings) and
[parquet_io.py](models/preprocessing/parquet_io.py) (reading and writing Parquet in batches).
The tokenizer has its own settings in [tokenizer_config.py](models/tokenizer/tokenizer_config.py)
(named so it never clashes with `config.py`) and reuses `parquet_io` and `light_cleaning` from preprocessing.

Every encoded article gets `<|bos|>` at the start and `<|eos|>` at the end. `<|pad|>` is only for batching.

The split files have the columns `title`, `text` and `language`. Training encodes `title\n\ntext` from the train
split into a compact cache, then shuffles and visits each 256-token window once per epoch. Validation uses its own
cache and is evaluated without gradients at each epoch's end. The `language` column remains available for later
per-language evaluation. The test split is not read.

## Tests

```bash
python models/preprocessing/tests/test_cleaning.py           # 12 cleaning cases (6 Arabic, 6 English), printed before / after
python models/preprocessing/tests/test_balance_and_split.py  # balancing and splitting on small fake data
python models/tokenizer/tests/test_tokenizer.py              # round trip, normalization of new text, special tokens
python models/gpt/tests/test_gpt.py                          # shapes, loss, causality, padding, tying and CPU overfit
python models/training/tests/test_training.py                # token cache, windows, accumulation, learning rate, exact resume, history
```

All also run with `pytest`.

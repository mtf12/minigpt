# miniGPT — Data Preparation & Tokenizer

> **🚧 Work in progress.** This project is under active development. The data preparation and tokenizer stages are complete;
> the next stages are still being built.
>
> - [x] Data preparation: cleaning, language balancing, train / val / test splits
> - [x] Tokenizer (trained on Arabic + English together)
> - [ ] Model (GPT, from scratch)
> - [ ] Training and evaluation (per language)

A bilingual (Arabic + English) GPT built from scratch. This part prepares the training data:
277,739 Wikipedia articles cleaned, balanced and split into a ~93M-token corpus, then a single 32k BPE tokenizer
trained on both languages together. Every decision was measured on the data.

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

## Results

| | Arabic | English |
|---|---|---|
| Raw articles | 35,952 | 241,787 |
| After cleaning | 30,733 (85.5%) | 184,332 (76.2%) |
| Estimated tokens | ~47.0M | ~46.3M |
| Train / val / test | 24,586 / 3,073 / 3,074 | 147,465 / 18,433 / 18,434 |

No article appears in two splits, and each split keeps the same Arabic / English ratio.

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
```

Or open the [notebook](models/data_pipeline.ipynb) and run the cells in order. The last cell encodes an Arabic and an
English sentence and decodes them back, so you can see the tokenizer working.
The raw data lives in `datasets/raw/ar.parquet` and `datasets/raw/en.parquet` and is never modified.

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

Shared files: [config.py](models/preprocessing/config.py) (paths and settings) and
[parquet_io.py](models/preprocessing/parquet_io.py) (reading and writing Parquet in batches).
The tokenizer has its own settings in [tokenizer_config.py](models/tokenizer/tokenizer_config.py)
(named so it never clashes with `config.py`) and reuses `parquet_io` and `light_cleaning` from preprocessing.

Every encoded article gets `<|bos|>` at the start and `<|eos|>` at the end. `<|pad|>` is only for batching.

The split files have the columns `title`, `text` and `language`. Training reads only `title` and `text`
(a list of `(title, text)` tuples); `language` is kept to evaluate each language separately.
Inside one language the articles keep their file order, so the training data loader shuffles them again.

## Tests

```bash
python models/preprocessing/tests/test_cleaning.py           # 12 cleaning cases (6 Arabic, 6 English), printed before / after
python models/preprocessing/tests/test_balance_and_split.py  # balancing and splitting on small fake data
python models/tokenizer/tests/test_tokenizer.py              # round trip, normalization of new text, special tokens
```

All also run with `pytest`.

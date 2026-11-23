# miniGPT — Data Preparation

> **🚧 Work in progress.** This project is under active development. The data preparation stage is complete;
> the next stages are still being built.
>
> - [x] Data preparation: cleaning, language balancing, train / val / test splits
> - [ ] Tokenizer (trained on Arabic + English together)
> - [ ] Model (GPT, from scratch)
> - [ ] Training and evaluation (per language)

A bilingual (Arabic + English) GPT built from scratch. This part prepares the training data:
277,739 Wikipedia articles cleaned, balanced and split into a ~93M-token corpus, with every decision measured on the data.

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

## Results

| | Arabic | English |
|---|---|---|
| Raw articles | 35,952 | 241,787 |
| After cleaning | 30,733 (85.5%) | 184,332 (76.2%) |
| Estimated tokens | ~47.0M | ~46.3M |
| Train / val / test | 24,586 / 3,073 / 3,074 | 147,465 / 18,433 / 18,434 |

No article appears in two splits, and each split keeps the same Arabic / English ratio.

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

# everything, with the reports
python models/preprocessing/run_preprocessing.py

# cleaning, balancing and splitting only
python models/preprocessing/run_preprocessing.py --skip-reports
```

Or open the [notebook](models/data_pipeline.ipynb) and run the cells in order.
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

Shared files: [config.py](models/preprocessing/config.py) (paths and settings) and
[parquet_io.py](models/preprocessing/parquet_io.py) (reading and writing Parquet in batches).

The split files have the columns `title`, `text` and `language`. Training reads only `title` and `text`
(a list of `(title, text)` tuples); `language` is kept to evaluate each language separately.
Inside one language the articles keep their file order, so the training data loader shuffles them again.

## Tests

```bash
python models/preprocessing/tests/test_cleaning.py           # 12 cleaning cases (6 Arabic, 6 English), printed before / after
python models/preprocessing/tests/test_balance_and_split.py  # balancing and splitting on small fake data
```

Both also run with `pytest`.

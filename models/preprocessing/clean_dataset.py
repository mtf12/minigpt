# Run the cleaning on the whole dataset in two steps:
#   1) light cleaning: raw/ -> cleaned/
#   2) heavy cleaning: cleaned/ -> cleaned/ (overwritten), then exact duplicates are dropped
# Articles shorter than MIN_CHARS are dropped after every step.
from collections import Counter
from multiprocessing import Pool
import hashlib

import config
import parquet_io
import light_cleaning
import heavy_cleaning

COLUMNS = ["id", "language", "title", "text"]


# Light cleaning of one raw article (runs in a worker process).
# Returns (article, None) or (None, reason it was dropped).
def light_clean_row(row):
    text = light_cleaning.normalize(row["text"] or "")
    if len(text) < config.MIN_CHARS:
        return None, "too short before cleaning"
    text = light_cleaning.clean_basic(text)
    if len(text) < config.MIN_CHARS:
        return None, "too short after basic cleaning"
    title = light_cleaning.normalize(row["title"] or "").strip()
    return {"id": str(row["id"]), "title": title, "text": text}, None


# Heavy cleaning of one light-cleaned article (runs in a worker process).
def heavy_clean_row(row):
    text = heavy_cleaning.clean_structure(row["text"])
    if len(text) < config.MIN_CHARS:
        return None, "too short after structural cleaning"
    return {"id": row["id"], "title": row["title"], "text": text}, None


# Clean one file batch by batch with `clean_row`, and write what is kept to `out_path`.
# Counts the dropped articles in `counts`; returns (articles read, articles kept).
def clean_file(in_path, out_path, lang, clean_row, counts, drop_duplicates=False):
    read = kept_total = 0
    seen = set()  # fingerprints of texts already kept (16 bytes each)
    writer = parquet_io.open_writer(out_path, COLUMNS)

    with Pool(config.WORKERS) as pool:
        for batch in parquet_io.read_batches(in_path):
            read += len(batch)
            kept = []
            for article, reason in pool.imap(clean_row, batch, chunksize=200):
                if article is None:
                    counts[reason] += 1
                    continue
                if drop_duplicates:
                    fingerprint = hashlib.blake2b(article["text"].encode(), digest_size=16).digest()
                    if fingerprint in seen:
                        counts["duplicates"] += 1
                        continue
                    seen.add(fingerprint)
                article["language"] = lang
                kept.append(article)
            parquet_io.write_batch(writer, kept)
            kept_total += len(kept)

    writer.close()
    return read, kept_total


# Step 1: light cleaning (normalization + basic) of raw/ into cleaned/. Returns what was removed and why.
def run_light_cleaning():
    stats = {}
    for lang in config.LANGUAGES:
        counts = Counter()
        read, kept = clean_file(config.RAW_DIR / f"{lang}.parquet", config.CLEANED_DIR / f"{lang}.parquet",
                                lang, light_clean_row, counts)
        counts["raw"] = read
        stats[lang] = counts
        print(f"{lang}: {read:,} raw -> {kept:,} after light cleaning")
    return stats


# Step 2: heavy cleaning (structural) of cleaned/, then drop exact duplicates.
# Takes the stats of step 1 and returns the full stats (used by the quality report).
def run_heavy_cleaning(light_stats):
    stats = {}
    for lang in config.LANGUAGES:
        counts = Counter(light_stats[lang])
        path = config.CLEANED_DIR / f"{lang}.parquet"
        temp_path = path.with_suffix(".tmp")
        read, kept = clean_file(path, temp_path, lang, heavy_clean_row, counts, drop_duplicates=True)
        temp_path.replace(path)
        counts["cleaned"] = kept
        stats[lang] = dict(counts)
        print(f"{lang}: {read:,} -> {kept:,} after heavy cleaning")
    return stats

# Split cleaned/ into train / val / test (80/10/10 per language, ar + en mixed) -> splits/.
from collections import Counter
import random

import pyarrow.parquet as pq

import config
import parquet_io


# Split each language 80/10/10 and mix ar + en in every split, batch by batch, into splits/.
# Columns: title, text, language. Training reads only title + text.
#
# How it works without loading everything: we walk through both languages at the same time.
# Each step picks a language at random (weighted by how many articles it has left), takes its
# next article, and sends it to train/val/test at random (weighted by how many each split still needs).
# This gives exact 80/10/10 per language and a random ar/en mix inside every split.
# Note: inside one language the articles keep their file order, so shuffle again when training.
def run():
    rng = random.Random(f"{config.SEED}-split")
    names = ["train", "val", "test"]
    readers, left, quota = {}, {}, {}
    for lang in config.LANGUAGES:
        path = config.CLEANED_DIR / f"{lang}.parquet"
        n = pq.ParquetFile(path).metadata.num_rows
        n_train, n_val = int(n * config.TRAIN_FRACTION), int(n * config.VAL_FRACTION)
        readers[lang] = parquet_io.read_rows(path, columns=["title", "text"])
        left[lang] = n
        quota[lang] = {"train": n_train, "val": n_val, "test": n - n_train - n_val}
        print(f"{lang}: train {n_train:,}, val {n_val:,}, test {n - n_train - n_val:,}")

    writers = {name: parquet_io.open_writer(config.SPLITS_DIR / f"{name}.parquet", ["title", "text", "language"])
               for name in names}
    buffers = {name: [] for name in names}
    counts = Counter()

    while sum(left.values()) > 0:
        lang = rng.choices(config.LANGUAGES, weights=[left[l] for l in config.LANGUAGES])[0]
        name = rng.choices(names, weights=[quota[lang][s] for s in names])[0]
        row = next(readers[lang])
        left[lang] -= 1
        quota[lang][name] -= 1

        buffers[name].append({"title": row["title"] or "", "text": row["text"], "language": lang})
        counts[name] += 1
        if len(buffers[name]) >= config.BATCH_SIZE:
            parquet_io.write_batch(writers[name], buffers[name])
            buffers[name] = []

    for name in names:
        parquet_io.write_batch(writers[name], buffers[name])
        writers[name].close()
        print(f"{name}: {counts[name]:,} articles")

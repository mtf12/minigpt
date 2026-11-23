# Balance Arabic and English by (estimated) tokens. The extra articles of the bigger language are deleted from cleaned/.
import random

import pyarrow.compute as pc
import pyarrow.parquet as pq

import config
import parquet_io


# Estimated tokens of every article in a file (reads only the text lengths, batch by batch).
def estimate_tokens(path, lang):
    tokens = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=config.BATCH_SIZE, columns=["text"]):
        tokens += [n / config.CHARS_PER_TOKEN[lang] for n in pc.utf8_length(batch["text"]).to_pylist()]
    return tokens


# Trim the bigger language (random articles) until both have about the same number of tokens.
def run():
    paths = {lang: config.CLEANED_DIR / f"{lang}.parquet" for lang in config.LANGUAGES}
    tokens = {lang: estimate_tokens(path, lang) for lang, path in paths.items()}
    target = min(sum(t) for t in tokens.values())

    for lang, path in paths.items():
        total = sum(tokens[lang])
        if total > target * (1 + config.BALANCE_TOLERANCE):
            # pick random articles until the target is reached
            order = list(range(len(tokens[lang])))
            random.Random(f"{config.SEED}-{lang}-balance").shuffle(order)
            keep, total = set(), 0
            for i in order:
                if total >= target:
                    break
                keep.add(i)
                total += tokens[lang][i]

            # copy the kept articles to a new file batch by batch, then replace the old one
            temp_path = path.with_suffix(".tmp")
            writer = parquet_io.open_writer(temp_path, ["id", "language", "title", "text"])
            index = 0
            for batch in parquet_io.read_batches(path):
                parquet_io.write_batch(writer, [row for i, row in enumerate(batch, index) if i in keep])
                index += len(batch)
            writer.close()
            temp_path.replace(path)
            tokens[lang] = [tokens[lang][i] for i in sorted(keep)]

        print(f"{lang}: {len(tokens[lang]):,} articles, ~{sum(tokens[lang]) / 1e6:.1f}M tokens")

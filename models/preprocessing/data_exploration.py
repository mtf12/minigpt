# Exploration report: size, lengths, problems, common lines, a sample article. Reads only.
from collections import Counter
import random
import re

import pyarrow.parquet as pq

import config
import parquet_io

# Things that should not be left in a clean article
LEFTOVER_HEADINGS = {
    "References", "See also", "External links", "Other websites", "Notes", "Sources",
    "مراجع", "المراجع", "وصلات خارجية", "روابط خارجية", "انظر أيضا", "انظر أيضاً", "انظر أيضًا", "مصادر",
}
URL = re.compile(r"(?:https?://|www\.)\S+")
MOJIBAKE = ["Ã©", "Ã¨", "Ã¡", "Ã³", "Ã¼", "Ã¶", "Ã¤", "Ä±", "â€", "Ø§", "Ù„"]


# Return the problems found in one article (leftover headings, tables, links, broken encoding).
def find_problems(text):
    problems = set()
    for line in text.split("\n"):
        line = line.strip()
        if line in LEFTOVER_HEADINGS:
            problems.add(f"leftover heading '{line}'")
        if line.startswith(("|", "{|")):
            problems.add("table markup")
        if line and sum(len(url) for url in URL.findall(line)) / len(line) >= 0.5:
            problems.add("link lines")
    if any(marker in text for marker in MOJIBAKE):
        problems.add("broken encoding")
    return problems


# Shorten a long article for printing: its start and its end.
def preview(text, head=600, tail=300):
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n   [...]\n" + text[-tail:]


# Pick `count` random row numbers of a file, without reading the file.
def random_rows(path, count):
    total = pq.ParquetFile(path).metadata.num_rows
    return set(random.Random(config.SEED).sample(range(total), min(count, total)))


# Exploration report for "raw" or "cleaned": size, lengths, problems, common short lines, a sample.
def explore(stage="raw"):
    folder = config.RAW_DIR if stage == "raw" else config.CLEANED_DIR
    tokens = {}
    for lang in config.LANGUAGES:
        path = folder / f"{lang}.parquet"
        sample_rows = random_rows(path, 1)
        lengths, size, sample = [], 0, None
        problems, short_lines = Counter(), Counter()

        for i, row in enumerate(parquet_io.read_rows(path)):
            text = row["text"] or ""
            lengths.append(len(text))
            size += len(text.encode())
            problems.update(find_problems(text))
            short_lines.update(line.strip() for line in text.split("\n") if 0 < len(line.strip()) <= 80)
            if i in sample_rows:
                sample = row

        lengths.sort()
        n = len(lengths)
        tokens[lang] = sum(lengths) / config.CHARS_PER_TOKEN[lang]

        print(f"\n===== {lang} / {stage} =====")
        print(f"articles: {n:,} | characters: {sum(lengths) / 1e6:.1f}M | size: {size / 1e6:.1f} MB")
        print(f"length: median {lengths[n // 2]:,} | p10 {lengths[n // 10]:,} | "
              f"p90 {lengths[n * 9 // 10]:,} | max {lengths[-1]:,}")
        print(f"shorter than {config.MIN_CHARS} chars: {sum(x < config.MIN_CHARS for x in lengths):,}")
        print("problems (articles):")
        for name, count in problems.most_common() or [("none", 0)]:
            print(f"  {name}: {count:,}")
        print("most common short lines:")
        for line, count in short_lines.most_common(10):
            print(f"  {count:7,}  {line}")
        print(f"random article: {sample['title']}\n{preview(sample['text'] or '')}")

    total = sum(tokens.values())
    print("\nestimated tokens: " + " | ".join(f"{lang} ~{t / 1e6:.1f}M ({t / total:.0%})" for lang, t in tokens.items()))

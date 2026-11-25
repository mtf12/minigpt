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


# Indent every line of a text, so a long article stays inside its section when printed.
def indent(text, spaces=4):
    return "\n".join(" " * spaces + line for line in text.split("\n"))


# Print one "label: value" line of a report, with the values of all lines lined up.
def print_row(label, value):
    print(f"  {label + ':':<18}{value}")


# Print a list under one label: the first item next to the label, the others lined up under it.
def print_list(label, items):
    items = list(items) or ["none"]
    print_row(label, items[0])
    for item in items[1:]:
        print(" " * 20 + item)


# Pick `count` random row numbers of a file, without reading the file.
def random_rows(path, count):
    total = pq.ParquetFile(path).metadata.num_rows
    return set(random.Random(config.SEED).sample(range(total), min(count, total)))


# Exploration report for "raw" or "cleaned": size, lengths, problems, common short lines, a sample.
def explore(stage="raw"):
    folder = config.RAW_DIR if stage == "raw" else config.CLEANED_DIR
    tokens = {}
    print(f"\n### 1) Exploration of the {stage} data")
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

        print(f"\n{config.LANGUAGE_NAMES[lang]}")
        print_row("articles", f"{n:,}")
        print_row("characters", f"{sum(lengths) / 1e6:.1f}M ({size / 1e6:.1f} MB)")
        print_row("article length", f"median {lengths[n // 2]:,} | p10 {lengths[n // 10]:,} | "
                                    f"p90 {lengths[n * 9 // 10]:,} | max {lengths[-1]:,}")
        print_row(f"under {config.MIN_CHARS} chars", f"{sum(x < config.MIN_CHARS for x in lengths):,} articles")
        print_list("problems", (f"{count:>7,}  {name}" for name, count in problems.most_common()))
        print_list("common lines", (f"{count:>7,}  {line}" for line, count in short_lines.most_common(10)))
        print_row("random article", sample["title"])
        print(indent(preview(sample["text"] or "")))

    total = sum(tokens.values())
    print("\nEstimated tokens")
    for lang, t in tokens.items():
        print_row(config.LANGUAGE_NAMES[lang], f"~{t / 1e6:.1f}M ({t / total:.0%})")

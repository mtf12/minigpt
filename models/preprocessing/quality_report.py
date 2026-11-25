# Quality report: raw vs cleaned, what was removed and why, leftover problems, before/after examples.
from collections import Counter

import config
import parquet_io
import data_exploration


# Compare raw with cleaned: what was removed and why, size, leftover problems, before/after examples.
def run(stats):
    print("\n### 4) Quality report: raw vs cleaned")
    for lang in config.LANGUAGES:
        raw_path = config.RAW_DIR / f"{lang}.parquet"
        clean_path = config.CLEANED_DIR / f"{lang}.parquet"

        # cleaned file: size, problems, and 2 random examples
        example_rows = data_exploration.random_rows(clean_path, 2)
        clean_size, problems, examples = 0, Counter(), []
        for i, row in enumerate(parquet_io.read_rows(clean_path)):
            clean_size += len(row["text"].encode())
            problems.update(data_exploration.find_problems(row["text"]))
            if i in example_rows:
                examples.append(row)

        # raw file: size, and the original text of the examples
        raw_size, originals = 0, {}
        example_ids = {row["id"] for row in examples}
        for row in parquet_io.read_rows(raw_path):
            raw_size += len((row["text"] or "").encode())
            if row["id"] in example_ids:
                originals[row["id"]] = row["text"]

        s = stats[lang]
        reasons = ["too short before cleaning", "too short after basic cleaning",
                   "too short after structural cleaning", "duplicates"]
        print(f"\n{config.LANGUAGE_NAMES[lang]}")
        data_exploration.print_row("articles", f"{s['raw']:,} -> {s['cleaned']:,} ({s['cleaned'] / s['raw']:.1%} kept)")
        data_exploration.print_list("removed", (f"{s.get(reason, 0):>7,}  {reason}" for reason in reasons))
        data_exploration.print_row("size", f"{raw_size / 1e6:.1f} MB -> {clean_size / 1e6:.1f} MB "
                                           f"({(clean_size - raw_size) / raw_size:+.1%})")
        data_exploration.print_list("problems left", (f"{count:>7,}  {name}" for name, count in problems.items()))

        for row in examples:
            before = originals[row["id"]]
            print(f"\n  Example: {row['title']} ({len(before):,} -> {len(row['text']):,} characters)")
            print(f"  before:\n{data_exploration.indent(data_exploration.preview(before))}")
            print(f"  after:\n{data_exploration.indent(data_exploration.preview(row['text']))}")

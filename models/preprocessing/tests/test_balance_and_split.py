from collections import Counter
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import balance_languages
import config
import parquet_io
import split_data

import pyarrow.parquet as pq

COLUMNS = ["id", "language", "title", "text"]


# Write a small fake cleaned/ folder: few long Arabic articles, many short English ones (like the real data).
def make_fake_cleaned(folder):
    articles = {
        "ar": [{"id": f"ar{i}", "language": "ar", "title": f"مقالة {i}", "text": f"نص عربي طويل رقم {i}. " * 20}
               for i in range(60)],
        "en": [{"id": f"en{i}", "language": "en", "title": f"Article {i}", "text": f"Short English text {i}. " * 4}
               for i in range(500)],
    }
    for lang, rows in articles.items():
        writer = parquet_io.open_writer(folder / f"{lang}.parquet", COLUMNS)
        parquet_io.write_batch(writer, rows)
        writer.close()


# Point config to a temp folder and use tiny batches, so the batching code is really exercised.
def use_temp_folders(root):
    config.CLEANED_DIR = root / "cleaned"
    config.SPLITS_DIR = root / "splits"
    config.BATCH_SIZE = 7


# Total estimated tokens of one cleaned language file.
def tokens(lang):
    return sum(balance_languages.estimate_tokens(config.CLEANED_DIR / f"{lang}.parquet", lang))


# Run balancing + splitting on fake data and check every rule; return the number of failures.
def run_checks():
    failures = 0

    # Print one check and count it if it failed.
    def check(name, passed, detail=""):
        nonlocal failures
        failures += not passed
        print(f"{'✅' if passed else '❌'} {name}" + (f"  ({detail})" if detail else ""))

    with tempfile.TemporaryDirectory() as tmp:
        use_temp_folders(Path(tmp))
        make_fake_cleaned(config.CLEANED_DIR)

        # ---------- balancing ----------
        before = {lang: tokens(lang) for lang in config.LANGUAGES}
        balance_languages.run()
        after = {lang: tokens(lang) for lang in config.LANGUAGES}
        ratio = max(after.values()) / min(after.values())
        check("[balance] the bigger language (English) was trimmed", after["en"] < before["en"],
              f"{before['en']:.0f} -> {after['en']:.0f} tokens")
        check("[balance] the smaller language (Arabic) was not touched", after["ar"] == before["ar"])
        check("[balance] both languages end up with about the same tokens", ratio < 1.05, f"ratio {ratio:.3f}")

        # ---------- splitting ----------
        cleaned = {}
        for lang in config.LANGUAGES:
            for row in parquet_io.read_rows(config.CLEANED_DIR / f"{lang}.parquet"):
                cleaned[row["text"]] = lang
        split_data.run()
        splits = {name: pq.read_table(config.SPLITS_DIR / f"{name}.parquet").to_pylist()
                  for name in ["train", "val", "test"]}
        texts = {name: {row["text"] for row in rows} for name, rows in splits.items()}

        overlap = (texts["train"] & texts["val"]) | (texts["train"] & texts["test"]) | (texts["val"] & texts["test"])
        check("[split] no article is in two splits", not overlap, f"{len(overlap)} shared")
        all_texts = texts["train"] | texts["val"] | texts["test"]
        check("[split] every cleaned article is in exactly one split, none lost",
              all_texts == set(cleaned) and sum(len(r) for r in splits.values()) == len(cleaned))
        check("[split] every row keeps its correct language", all(
            row["language"] == cleaned[row["text"]] for rows in splits.values() for row in rows))

        for lang in config.LANGUAGES:
            n = sum(1 for l in cleaned.values() if l == lang)
            counts = Counter(row["language"] for row in splits["train"])
            expected = int(n * config.TRAIN_FRACTION)
            check(f"[split] {lang}: train gets exactly {config.TRAIN_FRACTION:.0%}", counts[lang] == expected,
                  f"{counts[lang]} of {n}")
        check("[split] Arabic and English are mixed inside train",
              len({row["language"] for row in splits["train"][:50]}) == 2)
        check("[split] columns are title, text, language",
              pq.read_schema(config.SPLITS_DIR / "train.parquet").names == ["title", "text", "language"])

    print(f"\n{'all checks passed' if failures == 0 else f'{failures} check(s) failed'}")
    return failures


# Entry point for pytest: fails if any check fails.
def test_balance_and_split():
    assert run_checks() == 0


if __name__ == "__main__":
    sys.exit(1 if run_checks() else 0)

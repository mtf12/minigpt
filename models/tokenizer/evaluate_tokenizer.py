# Measure the tokenizer on the validation split: how many tokens each language needs per character.
import json
from tokenizers import Tokenizer
import pyarrow.parquet as pq
import tokenizer_config as config
import train_tokenizer


# Count articles, characters and tokens per language on the validation split (without BOS / EOS).
def measure(tokenizer):
    totals = {lang: {"articles": 0, "characters": 0, "tokens": 0} for lang in config.LANGUAGES}
    for article, lang in train_tokenizer.read_articles(config.VAL_PATH):
        totals[lang]["articles"] += 1
        totals[lang]["characters"] += len(article)
        totals[lang]["tokens"] += len(tokenizer.encode(article, add_special_tokens=False).ids)
    for counts in totals.values():
        counts["tokens_per_character"] = counts["tokens"] / counts["characters"]
    return {"vocab_size": tokenizer.get_vocab_size(), "languages": totals}


# Print the result of measure() as one table row per language.
def print_measure(result):
    print(f"  {'':<9}{'articles':>10}{'characters':>14}{'tokens':>12}{'chars / token':>16}")
    for lang, counts in result["languages"].items():
        print(f"  {config.LANGUAGE_NAMES[lang]:<9}{counts['articles']:>10,}{counts['characters']:>14,}"
              f"{counts['tokens']:>12,}{1 / counts['tokens_per_character']:>16.2f}")


# Measure the saved tokenizer (artifacts/tokenizer.json).
def evaluate_saved_tokenizer():
    print("\n### 8) The tokenizer on the validation split")
    result = measure(Tokenizer.from_file(str(config.TOKENIZER_PATH)))
    print_measure(result)
    return result


# Train one tokenizer per size in VOCAB_SIZE_CANDIDATES and compare them (slow).
# Saves the comparison to artifacts/vocab_benchmark.json; tokenizer.json is not changed.
def run_benchmark():
    print("\n### Comparing vocabulary sizes: validation tokens (one tokenizer trained per row, slow)")
    print(f"  {'vocabulary':>10}{'Arabic':>14}{'English':>14}")
    n = pq.ParquetFile(config.TRAIN_PATH).metadata.num_rows
    results = []
    for vocab_size in config.VOCAB_SIZE_CANDIDATES:
        texts = (article for article, _ in train_tokenizer.read_articles(config.TRAIN_PATH))
        result = measure(train_tokenizer.train_on_texts(texts, vocab_size, length=n))
        tokens = [result["languages"][lang]["tokens"] for lang in config.LANGUAGES]
        print(f"  {vocab_size:>10,}" + "".join(f"{count:>14,}" for count in tokens))
        results.append(result)

    config.VOCAB_BENCHMARK_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  saved to: {config.VOCAB_BENCHMARK_PATH.relative_to(config.PROJECT_ROOT)}")
    return results

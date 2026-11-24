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


# Measure the saved tokenizer (artifacts/tokenizer.json).
def evaluate_saved_tokenizer():
    result = measure(Tokenizer.from_file(str(config.TOKENIZER_PATH)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


# Train one tokenizer per size in VOCAB_SIZE_CANDIDATES and compare them (slow).
# Saves the comparison to artifacts/vocab_benchmark.json; tokenizer.json is not changed.
def run_benchmark():
    n = pq.ParquetFile(config.TRAIN_PATH).metadata.num_rows
    results = []
    for vocab_size in config.VOCAB_SIZE_CANDIDATES:
        print(f"\nTraining with a vocabulary of {vocab_size:,}")
        texts = (article for article, _ in train_tokenizer.read_articles(config.TRAIN_PATH))
        result = measure(train_tokenizer.train_on_texts(texts, vocab_size, length=n))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        results.append(result)

    config.VOCAB_BENCHMARK_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved the comparison to {config.VOCAB_BENCHMARK_PATH}")
    return results

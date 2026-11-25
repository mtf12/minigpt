# Run the tokenizer stage from the terminal, in the same order as the notebook (needs datasets/splits/):
#   python models/tokenizer/run_tokenizer.py
#   python models/tokenizer/run_tokenizer.py --benchmark   (also compare 16k / 24k / 32k / 48k vocabularies, slow)
import argparse
import evaluate_tokenizer
import train_tokenizer


# Train, then measure on validation; the vocabulary comparison is optional because it trains 4 tokenizers.
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the Arabic + English tokenizer.")
    parser.add_argument("--benchmark", action="store_true", help="also compare the vocabulary sizes in tokenizer_config")
    args = parser.parse_args()

    train_tokenizer.train()
    evaluate_tokenizer.evaluate_saved_tokenizer()
    if args.benchmark:
        evaluate_tokenizer.run_benchmark()


if __name__ == "__main__":
    main()

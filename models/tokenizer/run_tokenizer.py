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

    print("\n### 7) Training the tokenizer")
    train_tokenizer.train()

    print("\n### 8) Measuring it on the validation split")
    evaluate_tokenizer.evaluate_saved_tokenizer()

    if args.benchmark:
        print("\n### Optional) Comparing vocabulary sizes")
        evaluate_tokenizer.run_benchmark()


if __name__ == "__main__":
    main()

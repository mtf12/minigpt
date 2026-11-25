# Run the GPT training from the terminal, in the same order as the notebook (needs datasets/splits/):
#   python models/training/run_training.py
#   python models/training/run_training.py --resume                    (continue the last checkpoint, if any)
#   python models/training/run_training.py --epochs 1 --batch-size 64  (a shorter check)
import argparse

import training_config as config
import train_model


# Read the command line, then start or resume the training.
def main():
    parser = argparse.ArgumentParser(description="Train the Arabic + English GPT.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="number of passes over the train split")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="windows per update")
    parser.add_argument("--resume", action="store_true", help="continue the last checkpoint, if there is one")
    args = parser.parse_args()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    train_model.run_training(resume=args.resume)


if __name__ == "__main__":
    main()

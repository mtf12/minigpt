# After training: draw the loss curves from history.json and generate text with the best model.
import json
import sys

import matplotlib.pyplot as plt
import torch

import training_config as config
import encode_text
import gpt_model
import run_gpt
import train_model

TRAIN_COLOR = "#2a78d6"  # blue, first series color (checked for color-blind readers together with the orange)
VALIDATION_COLOR = "#eb6834"  # orange, second series color
MUTED_INK = "#898781"  # axes, ticks and grid stay quiet so the two curves stand out


# Train loss of every update (thin line) and validation loss after every epoch (dots), on one loss axis.
def plot_history(path=config.HISTORY_PATH):
    print("\n### 12) Loss curves")
    history = json.loads(path.read_text(encoding="utf-8"))
    updates, epochs = history["updates"], history["epochs"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot([u["step"] for u in updates], [u["loss"] for u in updates], color=TRAIN_COLOR, linewidth=1.2,
            label="train (every update)")
    ax.plot([e["step"] for e in epochs], [e["validation"]["loss"] for e in epochs], color=VALIDATION_COLOR,
            linewidth=2, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2,
            label="validation (after every epoch)")
    for e in epochs:
        ax.annotate(f"{e['validation']['loss']:.2f}", (e["step"], e["validation"]["loss"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#52514e")

    ax.set_title(f"Loss over {len(updates):,} updates and {len(epochs)} epochs", loc="left", fontsize=12)
    ax.set_xlabel("update", color=MUTED_INK)
    ax.set_ylabel("cross-entropy loss", color=MUTED_INK)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.tick_params(colors=MUTED_INK)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    plt.show()


# Load the best model (lowest validation loss) with the tokenizer it was trained with, and continue every prompt.
def generate_samples(prompts):
    print("\n### 13) Text from the best model")
    device = gpt_model.select_device()
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=True)
    model = gpt_model.GPTModel().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    progress = checkpoint["progress"]
    if train_model.tokenizer_fingerprint(config.SAVED_TOKENIZER_PATH) != progress["tokenizer"]:
        sys.exit(f"{config.SAVED_TOKENIZER_PATH} is not the tokenizer {config.BEST_MODEL_PATH.name} was trained with.")
    print(f"  {'model:':<12}epoch {progress['best_epoch']}, validation loss {progress['best_val_loss']:.4f}")
    print(f"  {'tokenizer:':<12}{config.SAVED_TOKENIZER_PATH} (the one the model was trained with)")

    torch.manual_seed(config.SEED)
    tokenizer = encode_text.load_tokenizer(config.SAVED_TOKENIZER_PATH)
    for prompt in prompts:
        ids = model.generate(run_gpt.prompt_ids(tokenizer, prompt, device), max_new_tokens=config.SAMPLE_TOKENS)
        print(f"\n  {'prompt:':<12}{prompt}")
        print(f"  {'sample:':<12}{encode_text.decode(tokenizer, ids[0].tolist())}")

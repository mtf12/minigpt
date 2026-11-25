# Train the GPT on the train split epoch by epoch, check the validation loss after every epoch, and save checkpoints.
# The last checkpoint is saved every CHECKPOINT_MINUTES and after every epoch, so a dropped session can resume.
import hashlib
import json
import math
import shutil
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

import training_config as config
import encode_text
import gpt_model
import train_tokenizer


# ---------------------------------------------------------------------------
# Data: token caches and windows
# ---------------------------------------------------------------------------

# Encode every article of a split once and save all token ids in one uint16 file (the 32,000 ids fit in 16 bits).
# The file is reused while it is newer than the split and the tokenizer.
def build_token_cache(split_path, cache_path, tokenizer):
    newest_source = max(split_path.stat().st_mtime, config.TOKENIZER_PATH.stat().st_mtime)
    if not cache_path.exists() or cache_path.stat().st_mtime < newest_source:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")  # a session dropped mid-way never leaves half a cache
        with open(temp_path, "wb") as f:
            buffer = []
            for article, _ in train_tokenizer.read_articles(split_path):
                buffer += encode_text.encode(tokenizer, article)
                if len(buffer) >= config.CACHE_BUFFER_TOKENS:
                    np.array(buffer, dtype=np.uint16).tofile(f)
                    buffer = []
            np.array(buffer, dtype=np.uint16).tofile(f)
        temp_path.replace(cache_path)

    return np.memmap(cache_path, dtype=np.uint16, mode="r")  # read from disk, never fully in memory


# Number of context windows in a token stream; the last one can be shorter (it gets padded).
def count_windows(tokens):
    return math.ceil((len(tokens) - 1) / config.CONTEXT_LENGTH)


# The order of the windows in one epoch: shuffled, every window once, and the same order after a resume.
def epoch_order(tokens, epoch):
    return np.random.default_rng(config.SEED + epoch).permutation(count_windows(tokens))


# Inputs and targets (the inputs shifted by one token) of some windows, padded at the end of a short last window.
def get_batch(tokens, windows, device):
    ids = np.full((len(windows), config.CONTEXT_LENGTH), config.PAD_ID, dtype=np.int64)
    targets = np.full_like(ids, config.PAD_ID)
    for row, window in enumerate(windows):
        start = int(window) * config.CONTEXT_LENGTH
        length = min(config.CONTEXT_LENGTH, len(tokens) - 1 - start)
        ids[row, :length] = tokens[start : start + length]
        targets[row, :length] = tokens[start + 1 : start + length + 1]
    return torch.from_numpy(ids).to(device), torch.from_numpy(targets).to(device)


# Number of real (not padded) targets in some windows.
def count_targets(tokens, windows):
    return sum(min(config.CONTEXT_LENGTH, len(tokens) - 1 - int(window) * config.CONTEXT_LENGTH) for window in windows)


# ---------------------------------------------------------------------------
# One update and one evaluation
# ---------------------------------------------------------------------------

# Learning rate of an update: grows linearly for WARMUP_STEPS, then follows a cosine down to MIN_LEARNING_RATE.
def learning_rate(step, total_steps):
    if step < config.WARMUP_STEPS:
        return config.LEARNING_RATE * (step + 1) / config.WARMUP_STEPS
    progress = (step - config.WARMUP_STEPS) / max(total_steps - config.WARMUP_STEPS, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return config.MIN_LEARNING_RATE + (config.LEARNING_RATE - config.MIN_LEARNING_RATE) * cosine


# Mixed precision (float16) on CUDA only; MPS and CPU use normal float32.
def autocast(device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda")


# Loss sum, correct next-token guesses and real targets of one forward pass.
def batch_sums(logits, loss, targets):
    real = targets != config.PAD_ID
    count = int(real.sum())
    return loss.item() * count, int(((logits.argmax(dim=-1) == targets) & real).sum()), count


# Turn the sums of an epoch into loss, accuracy and perplexity.
def summarize(loss_sum, correct, count):
    loss = loss_sum / count
    return {"loss": loss, "accuracy": correct / count, "perplexity": math.exp(loss)}


# One AdamW update on a batch of windows: add up the gradients of its micro-batches, clip them, then step.
# Returns the loss sum, correct guesses and real targets of the batch.
def train_step(run, tokens, windows, lr):
    model, optimizer, scaler, device = run["model"], run["optimizer"], run["scaler"], run["device"]
    model.train()
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)

    total = count_targets(tokens, windows)
    sums = [0.0, 0, 0]
    for start in range(0, len(windows), config.MICRO_BATCH_SIZE):
        ids, targets = get_batch(tokens, windows[start : start + config.MICRO_BATCH_SIZE], device)
        with autocast(device):
            logits, loss = model(ids, targets)
        loss_sum, correct, count = batch_sums(logits, loss, targets)
        scaler.scale(loss * count / total).backward()  # weighted, so the batch loss is the mean over all its tokens
        sums = [sums[0] + loss_sum, sums[1] + correct, sums[2] + count]

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    return sums


# Loss, accuracy and perplexity on every window of a split, without gradients.
def evaluate(run, tokens):
    model, device = run["model"], run["device"]
    model.eval()
    sums = [0.0, 0, 0]
    windows = count_windows(tokens)
    with torch.no_grad():
        for start in range(0, windows, config.MICRO_BATCH_SIZE):
            ids, targets = get_batch(tokens, range(start, min(start + config.MICRO_BATCH_SIZE, windows)), device)
            with autocast(device):
                logits, loss = model(ids, targets)
            loss_sum, correct, count = batch_sums(logits, loss, targets)
            sums = [sums[0] + loss_sum, sums[1] + correct, sums[2] + count]
    return summarize(*sums)


# ---------------------------------------------------------------------------
# The run and its checkpoints
# ---------------------------------------------------------------------------

# Short fingerprint of a tokenizer.json: a checkpoint must always be used with the tokenizer it was trained with.
def tokenizer_fingerprint(path=None):
    return hashlib.sha256((path or config.TOKENIZER_PATH).read_bytes()).hexdigest()[:16]


# Everything a training run needs, and everything a checkpoint saves: model, optimizer, scaler and progress.
def new_run(model, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2),
                                  weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")  # does nothing when not on CUDA
    progress = {
        "epoch": 0, "step": 0, "global_step": 0,          # where training is
        "loss_sum": 0.0, "correct": 0, "targets": 0,      # sums of the current epoch, for the train metrics
        "best_val_loss": float("inf"), "best_epoch": 0,
        "history": {"updates": [], "epochs": []},        # train loss of every update, metrics of every epoch
        "tokenizer": tokenizer_fingerprint(),
    }
    return {"model": model, "optimizer": optimizer, "scaler": scaler, "device": device, "progress": progress}


# Save a run through a temporary file, so a dropped session never leaves a broken checkpoint.
def save_checkpoint(path, run):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    torch.save({
        "model": run["model"].state_dict(),
        "optimizer": run["optimizer"].state_dict(),
        "scaler": run["scaler"].state_dict(),
        "progress": run["progress"],
        "rng": torch.get_rng_state(),
    }, temp_path)
    temp_path.replace(path)


# Save the last checkpoint and write the history next to it (history.json is what the loss curve is drawn from).
# Both always describe the same update, so after a resume the history continues without gaps or repeats.
# The tokenizer of the run is copied next to them once, so the checkpoints can always be used with the right one.
def save_progress(run):
    save_checkpoint(config.LAST_CHECKPOINT_PATH, run)
    config.HISTORY_PATH.write_text(json.dumps(run["progress"]["history"], indent=1) + "\n", encoding="utf-8")
    saved = config.SAVED_TOKENIZER_PATH
    if not saved.exists() or tokenizer_fingerprint(saved) != run["progress"]["tokenizer"]:
        shutil.copy2(config.TOKENIZER_PATH, saved)


# Load a checkpoint into a run, so training continues at the exact next update.
def load_checkpoint(path, run):
    checkpoint = torch.load(path, map_location=run["device"], weights_only=True)
    if checkpoint["progress"]["tokenizer"] != tokenizer_fingerprint():
        sys.exit(f"{path.name} was trained with another tokenizer.json. Run the notebook from the top "
                 "(stages 1-7 rebuild the same splits and tokenizer) before resuming.")
    run["model"].load_state_dict(checkpoint["model"])
    run["optimizer"].load_state_dict(checkpoint["optimizer"])
    if run["scaler"].is_enabled() and checkpoint["scaler"]:  # the scaler is only used (and saved) on CUDA
        run["scaler"].load_state_dict(checkpoint["scaler"])
    run["progress"] = checkpoint["progress"]
    torch.set_rng_state(checkpoint["rng"].cpu())
    print(f"  {'resumed from:':<15}{path.name}, epoch {run['progress']['epoch'] + 1}, update {run['progress']['step']:,}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# Number of updates in one epoch.
def steps_per_epoch(tokens):
    return math.ceil(count_windows(tokens) / config.BATCH_SIZE)


# Train the rest of the current epoch and save the last checkpoint every CHECKPOINT_MINUTES.
def train_epoch(run, tokens, total_steps):
    progress = run["progress"]
    order = epoch_order(tokens, progress["epoch"])
    steps = steps_per_epoch(tokens)
    last_save = time.time()

    bar = tqdm(range(progress["step"], steps), desc=f"epoch {progress['epoch'] + 1}/{config.EPOCHS}",
               initial=progress["step"], total=steps)
    for step in bar:
        windows = order[step * config.BATCH_SIZE : (step + 1) * config.BATCH_SIZE]
        lr = learning_rate(progress["global_step"], total_steps)
        loss_sum, correct, count = train_step(run, tokens, windows, lr)
        progress["loss_sum"] += loss_sum
        progress["correct"] += correct
        progress["targets"] += count
        progress["step"] = step + 1
        progress["global_step"] += 1
        progress["history"]["updates"].append({"step": progress["global_step"], "loss": loss_sum / count, "lr": lr})
        bar.set_postfix(loss=f"{loss_sum / count:.4f}")

        if time.time() - last_save >= config.CHECKPOINT_MINUTES * 60:
            save_progress(run)
            last_save = time.time()


# Print the train and validation metrics of one epoch.
def print_epoch(entry):
    title = f"Epoch {entry['epoch']}/{config.EPOCHS}"
    print(f"\n  {title:<12}{'loss':>12}{'accuracy':>12}{'perplexity':>13}")
    for name in ["train", "validation"]:
        metrics = entry[name]
        print(f"  {name:<12}{metrics['loss']:>12.4f}{metrics['accuracy']:>12.2%}{metrics['perplexity']:>13.2f}")


# Train until EPOCHS from wherever the run is. After each epoch: validation, best model, last checkpoint.
def fit(run, train_tokens, val_tokens):
    progress = run["progress"]
    total_steps = config.EPOCHS * steps_per_epoch(train_tokens)
    while progress["epoch"] < config.EPOCHS:
        train_epoch(run, train_tokens, total_steps)
        entry = {
            "epoch": progress["epoch"] + 1,
            "step": progress["global_step"],
            "train": summarize(progress["loss_sum"], progress["correct"], progress["targets"]),
            "validation": evaluate(run, val_tokens),
        }
        print_epoch(entry)
        progress["history"]["epochs"].append(entry)
        progress.update(epoch=progress["epoch"] + 1, step=0, loss_sum=0.0, correct=0, targets=0)

        if entry["validation"]["loss"] < progress["best_val_loss"]:
            progress["best_val_loss"] = entry["validation"]["loss"]
            progress["best_epoch"] = entry["epoch"]
            save_checkpoint(config.BEST_MODEL_PATH, run)
        save_progress(run)
    return progress["history"]["epochs"]


# Build the token caches, then start a new run (or, with resume, continue the last checkpoint if there is one).
def run_training(resume=False):
    print("\n### 11) Training the GPT")
    torch.manual_seed(config.SEED)
    device = gpt_model.select_device()
    tokenizer = encode_text.load_tokenizer()
    train_tokens = build_token_cache(config.TRAIN_PATH, config.TRAIN_TOKENS_PATH, tokenizer)
    val_tokens = build_token_cache(config.VAL_PATH, config.VAL_TOKENS_PATH, tokenizer)

    run = new_run(gpt_model.GPTModel().to(device), device)
    if resume and config.LAST_CHECKPOINT_PATH.exists():
        load_checkpoint(config.LAST_CHECKPOINT_PATH, run)

    steps = steps_per_epoch(train_tokens)
    print(f"  {'device:':<15}{device}")
    print(f"  {'tokens:':<15}{len(train_tokens):,} train | {len(val_tokens):,} validation")
    print(f"  {'schedule:':<15}{config.EPOCHS} epochs x {steps:,} updates of {config.BATCH_SIZE} windows "
          f"({config.BATCH_SIZE // config.MICRO_BATCH_SIZE} micro-batches of {config.MICRO_BATCH_SIZE})")
    print(f"  {'checkpoints:':<15}{config.CHECKPOINTS_DIR}\n")
    history = fit(run, train_tokens, val_tokens)
    print(f"\n  best validation loss: {run['progress']['best_val_loss']:.4f} at epoch {run['progress']['best_epoch']}")
    return history

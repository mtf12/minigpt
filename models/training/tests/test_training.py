from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import training_config as config
import encode_text
import gpt_model
import train_model

DEVICE = torch.device("cpu")
TOKENS = (np.arange(65) % 28 + 4).astype(np.uint16)  # 65 fake token ids (no special ids): 8 windows of 8 tokens


# Point config to a temp folder and use tiny sizes, so a whole training run takes a second on CPU.
def use_tiny_settings(folder):
    config.CONTEXT_LENGTH = 8
    config.BATCH_SIZE = 2
    config.MICRO_BATCH_SIZE = 1
    config.EPOCHS = 2
    config.WARMUP_STEPS = 2
    config.CHECKPOINT_MINUTES = 0  # save the last checkpoint after every update
    config.LAST_CHECKPOINT_PATH = folder / "last_checkpoint.pt"
    config.BEST_MODEL_PATH = folder / "best_model.pt"
    config.HISTORY_PATH = folder / "history.json"
    config.SAVED_TOKENIZER_PATH = folder / "tokenizer.json"


# A new run with a tiny GPT; the same seed always gives the same starting weights.
def tiny_run():
    torch.manual_seed(42)
    model = gpt_model.GPTModel(vocab_size=32, n_layer=1, n_head=2, d_model=16, context_length=8, dropout=0.0)
    return train_model.new_run(model, DEVICE)


# All weights of two models are the same.
def same_weights(first, second):
    return all(torch.allclose(a, b, atol=1e-6) for a, b in zip(first.parameters(), second.parameters()))


# Run every check on tiny data, print them, and return the number of failures.
def run_checks():
    failures = 0

    # Print one check and count it if it failed.
    def check(name, passed, detail=""):
        nonlocal failures
        failures += not passed
        print(f"{'✅' if passed else '❌'} {name}" + (f"  ({detail})" if detail else ""))

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)

        # ---------- data ----------
        rows = [{"title": "Cairo", "text": "A city in Egypt.", "language": "en"},
                {"title": "القاهرة", "text": "مدينة في مصر.", "language": "ar"}]
        pq.write_table(pa.Table.from_pylist(rows), folder / "train.parquet")
        tokenizer = encode_text.load_tokenizer()
        tokens = train_model.build_token_cache(folder / "train.parquet", folder / "train_tokens.u16", tokenizer)
        expected = [i for row in rows for i in encode_text.encode(tokenizer, f"{row['title']}\n\n{row['text']}")]
        check("[data] the cache holds every article encoded with BOS / EOS", tokens.tolist() == expected,
              f"{len(tokens)} tokens")
        written = (folder / "train_tokens.u16").stat().st_mtime_ns
        train_model.build_token_cache(folder / "train.parquet", folder / "train_tokens.u16", tokenizer)
        check("[data] a cache newer than its split is reused, not rebuilt",
              (folder / "train_tokens.u16").stat().st_mtime_ns == written)

        use_tiny_settings(folder)
        order = train_model.epoch_order(TOKENS, epoch=0)
        check("[data] an epoch visits every window once, in a shuffled order",
              sorted(order.tolist()) == list(range(8)) and order.tolist() != list(range(8)), f"{order.tolist()}")
        check("[data] the order of an epoch is the same after a resume",
              np.array_equal(order, train_model.epoch_order(TOKENS, epoch=0)))
        ids, targets = train_model.get_batch(np.arange(4, 24, dtype=np.uint16), [0, 2], DEVICE)
        check("[data] targets are the inputs shifted by one token", torch.equal(ids[0, 1:], targets[0, :-1]))
        check("[data] a short last window is padded", ids[1, 3:].eq(config.PAD_ID).all().item()
              and targets[1, 3:].eq(config.PAD_ID).all().item(), f"{ids[1].tolist()}")

        # ---------- one update ----------
        whole, split = tiny_run(), tiny_run()
        config.MICRO_BATCH_SIZE = 4
        train_model.train_step(whole, TOKENS, np.array([0, 1, 2, 7]), lr=0.01)
        config.MICRO_BATCH_SIZE = 1
        train_model.train_step(split, TOKENS, np.array([0, 1, 2, 7]), lr=0.01)
        gaps = [(a.grad - b.grad).abs().max().item() for a, b in zip(whole["model"].parameters(), split["model"].parameters())]
        check("[update] 4 micro-batches of 1 give the same gradients as one batch of 4", max(gaps) < 1e-6,
              f"largest difference {max(gaps):.0e}")

        config.WARMUP_STEPS = 100
        rates = [train_model.learning_rate(step, 1641) for step in range(1641)]
        check("[update] the learning rate warms up to its peak in 100 updates",
              rates[0] < rates[50] < rates[99] == config.LEARNING_RATE, f"{rates[0]:.1e} -> {rates[99]:.1e}")
        check("[update] then it goes down (cosine) to the minimum at the last update",
              all(a >= b for a, b in zip(rates[99:], rates[100:])) and abs(rates[-1] - config.MIN_LEARNING_RATE) < 1e-6,
              f"last {rates[-1]:.1e}")
        config.WARMUP_STEPS = 2

        with patch("torch.cuda.is_available", return_value=True):
            check("[device] CUDA is picked first (Colab)", gpt_model.select_device().type == "cuda")

        # ---------- epochs and checkpoints ----------
        straight = tiny_run()
        history = train_model.fit(straight, TOKENS, TOKENS[:33])
        check("[fit] every epoch reports train and validation metrics",
              len(history) == 2 and all("perplexity" in entry["validation"] for entry in history),
              f"validation loss {history[0]['validation']['loss']:.3f} -> {history[1]['validation']['loss']:.3f}")
        check("[fit] the last checkpoint and the best model are saved",
              config.LAST_CHECKPOINT_PATH.exists() and config.BEST_MODEL_PATH.exists())
        check("[fit] the tokenizer of the run is saved next to the checkpoints",
              config.SAVED_TOKENIZER_PATH.read_bytes() == config.TOKENIZER_PATH.read_bytes())
        straight_history = json.loads(config.HISTORY_PATH.read_text())
        check("[fit] history.json has the loss of every update and every epoch",
              [u["step"] for u in straight_history["updates"]] == list(range(1, 9)) and len(straight_history["epochs"]) == 2,
              f"{len(straight_history['updates'])} updates, {len(straight_history['epochs'])} epochs")

        # a session that drops after 5 updates (in the middle of epoch 2), then resumes from the last checkpoint
        dropped = tiny_run()
        original_step = train_model.train_step
        calls = []

        # The normal update, but the "session" dies before the 6th one.
        def dropping_step(*args):
            calls.append(1)
            if len(calls) > 5:
                raise SystemExit("session dropped")
            return original_step(*args)

        with patch.object(train_model, "train_step", dropping_step):
            try:
                train_model.fit(dropped, TOKENS, TOKENS[:33])
            except SystemExit:
                pass
        resumed = tiny_run()
        train_model.load_checkpoint(config.LAST_CHECKPOINT_PATH, resumed)
        position = (resumed["progress"]["epoch"], resumed["progress"]["step"])
        train_model.fit(resumed, TOKENS, TOKENS[:33])
        check("[resume] the checkpoint is at the last finished update", position == (1, 1), f"epoch, update = {position}")
        check("[resume] resuming ends with the same weights as training without a stop",
              same_weights(straight["model"], resumed["model"]))
        check("[resume] and with the same metrics",
              abs(straight["progress"]["history"]["epochs"][-1]["train"]["loss"]
                  - resumed["progress"]["history"]["epochs"][-1]["train"]["loss"]) < 1e-6)
        resumed_history = json.loads(config.HISTORY_PATH.read_text())
        check("[resume] history.json keeps every update once, with no gap or repeat after the resume",
              [u["step"] for u in resumed_history["updates"]] == list(range(1, 9))
              and all(abs(a["loss"] - b["loss"]) < 1e-6 for a, b in zip(straight_history["updates"], resumed_history["updates"])))

        # the same checkpoint with a tokenizer.json that changed in between
        real_tokenizer = config.TOKENIZER_PATH
        config.TOKENIZER_PATH = folder / "changed_tokenizer.json"
        config.TOKENIZER_PATH.write_text(real_tokenizer.read_text(encoding="utf-8") + " ", encoding="utf-8")
        try:
            train_model.load_checkpoint(config.LAST_CHECKPOINT_PATH, tiny_run())
            refused = False
        except SystemExit:
            refused = True
        config.TOKENIZER_PATH = real_tokenizer
        check("[resume] a checkpoint refuses to continue with a different tokenizer", refused)

    print(f"\n{'all checks passed' if failures == 0 else f'{failures} check(s) failed'}")
    return failures


# Entry point for pytest: fails if any check fails.
def test_training():
    assert run_checks() == 0


if __name__ == "__main__":
    sys.exit(1 if run_checks() else 0)

from pathlib import Path
import math
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_config as config
import gpt_model


CASES = [
    "logits have batch, sequence and vocabulary dimensions",
    "initial loss is close to log vocabulary size",
    "future tokens do not change earlier logits",
    "padding targets are ignored by cross-entropy",
    "the output layer shares the token embedding weights",
    "a tiny model can overfit one batch on CPU",
    "generation stops after EOS",
]


# Print one result and return whether the check failed.
def check(name, passed, detail=""):
    print(f"{'✅' if passed else '❌'} {name}" + (f"  ({detail})" if detail else ""))
    return not passed


# Run model shape, initialization, causality, padding, tying and overfit checks.
def run_cases():
    failures = 0
    torch.manual_seed(42)
    model = gpt_model.GPTModel(dropout=0.0).cpu().eval()
    ids = torch.randint(4, config.VOCAB_SIZE, (4, 16))
    targets = torch.randint(4, config.VOCAB_SIZE, (4, 16))
    logits, loss = model(ids, targets)

    failures += check(
        CASES[0],
        tuple(logits.shape) == (4, 16, config.VOCAB_SIZE),
        f"got {tuple(logits.shape)}",
    )
    expected_loss = math.log(config.VOCAB_SIZE)
    failures += check(
        CASES[1],
        abs(loss.item() - expected_loss) < 0.5,
        f"got {loss.item():.4f}, expected about {expected_loss:.4f}",
    )

    tiny = gpt_model.GPTModel(
        vocab_size=32,
        n_layer=1,
        n_head=2,
        d_model=16,
        context_length=8,
        dropout=0.0,
    ).cpu().eval()
    causal_ids = torch.tensor([[config.BOS_ID, 4, 5, 6, 7, 8]])
    changed_ids = causal_ids.clone()
    changed_ids[0, 5] = 9
    original_logits, _ = tiny(causal_ids)
    changed_logits, _ = tiny(changed_ids)
    failures += check(
        CASES[2],
        torch.allclose(original_logits[:, :5], changed_logits[:, :5], atol=1e-6),
    )

    pad_ids = torch.tensor([[config.BOS_ID, 4, 5], [config.BOS_ID, 6, 7]])
    pad_targets = torch.tensor([[4, 5, config.PAD_ID], [6, config.PAD_ID, config.PAD_ID]])
    pad_logits, pad_loss = tiny(pad_ids, pad_targets)
    expected_pad_loss = F.cross_entropy(
        pad_logits.reshape(-1, tiny.vocab_size),
        pad_targets.reshape(-1),
        ignore_index=config.PAD_ID,
    )
    failures += check(CASES[3], torch.allclose(pad_loss, expected_pad_loss))
    failures += check(
        CASES[4],
        model.lm_head.weight is model.token_embedding.weight,
        f"{model.count_parameters()[0]:,} unique parameters",
    )

    torch.manual_seed(42)
    overfit_model = gpt_model.GPTModel(
        vocab_size=16,
        n_layer=1,
        n_head=2,
        d_model=16,
        context_length=8,
        dropout=0.0,
    ).cpu().train()
    sequence = torch.tensor([[config.BOS_ID, 4, 5, 6, 7, config.EOS_ID]])
    train_ids, train_targets = sequence[:, :-1], sequence[:, 1:]
    optimizer = torch.optim.AdamW(overfit_model.parameters(), lr=0.03, weight_decay=0.0)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        _, train_loss = overfit_model(train_ids, train_targets)
        train_loss.backward()
        optimizer.step()
    failures += check(CASES[5], train_loss.item() < 0.1, f"loss {train_loss.item():.4f} after 300 steps")

    eos_model = gpt_model.GPTModel(
        vocab_size=16,
        n_layer=1,
        n_head=2,
        d_model=16,
        context_length=8,
        dropout=0.0,
    ).cpu().eval()
    with torch.no_grad():
        for parameter in eos_model.parameters():
            parameter.zero_()
        eos_model.final_norm.weight.fill_(1.0)
        eos_model.position_embedding.weight[0, 0] = 1.0
        eos_model.token_embedding.weight[config.EOS_ID, 0] = 1.0
    generated = eos_model.generate(torch.tensor([[config.BOS_ID]]), max_new_tokens=5, top_k=1)
    failures += check(
        CASES[6],
        len(generated[0]) == 2 and generated[0, -1].item() == config.EOS_ID,
        f"generated ids {generated[0].tolist()}",
    )

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return failures


# Entry point for pytest.
def test_gpt_cases():
    assert run_cases() == 0


if __name__ == "__main__":
    sys.exit(1 if run_cases() else 0)
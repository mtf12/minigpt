# Run the GPT model on a validation batch and generate short Arabic and English samples.
#   python models/gpt/run_gpt.py
import torch

import gpt_config as config
import encode_text
import gpt_model
import train_tokenizer


# One validation article per language as a batch: inputs and targets (shifted by one token), padded to one length.
def read_batch(tokenizer, device):
    sequences = {}
    for article, lang in train_tokenizer.read_articles(config.VAL_PATH):
        if lang not in sequences:
            sequences[lang] = encode_text.encode(tokenizer, article)[: config.CONTEXT_LENGTH + 1]
        if len(sequences) == config.DEMO_BATCH_SIZE:
            break

    length = max(len(sequence) for sequence in sequences.values()) - 1
    ids = torch.full((config.DEMO_BATCH_SIZE, length), config.PAD_ID)
    targets = torch.full_like(ids, config.PAD_ID)
    for row, lang in enumerate(["ar", "en"]):
        sequence = torch.tensor(sequences[lang])
        ids[row, : len(sequence) - 1] = sequence[:-1]
        targets[row, : len(sequence) - 1] = sequence[1:]
    return ids.to(device), targets.to(device)


# Encode a prompt without EOS and add BOS before asking the model to continue it.
def prompt_ids(tokenizer, prompt, device):
    ids = encode_text.encode(tokenizer, prompt, add_bos_eos=False)
    return torch.tensor([[config.BOS_ID] + ids], device=device)


# Build the model, measure its size and loss, then generate one sample per language.
def main():
    torch.manual_seed(config.SEED)
    device = gpt_model.select_device()
    tokenizer = encode_text.load_tokenizer()
    model = gpt_model.GPTModel().to(device)
    model.eval()

    total, non_embedding = model.count_parameters()
    print("\n### 10) GPT model")
    print(f"device: {device}")
    print(f"parameters: {total:,} total, {non_embedding:,} non-embedding, {total - non_embedding:,} embedding")

    ids, targets = read_batch(tokenizer, device)
    _, loss = model(ids, targets)
    print(f"validation batch: {tuple(ids.shape)}, loss at init: {loss.item():.4f}")

    for language, prompt in [("Arabic", "القاهرة هي عاصمة مصر"), ("English", "Cairo is the capital of")]:
        generated = model.generate(prompt_ids(tokenizer, prompt, device))
        print(f"\n{language} prompt: {prompt}")
        print(f"sample: {encode_text.decode(tokenizer, generated[0].tolist())}")


if __name__ == "__main__":
    main()

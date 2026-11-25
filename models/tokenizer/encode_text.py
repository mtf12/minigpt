# Use the trained tokenizer on new text (a user prompt, new data).
# New text did not go through the preprocessing, so it gets the same normalization as the training data first
# (NFC, invisible characters and stretching tatweel removed), otherwise the model sees forms it never saw in training.
from tokenizers import Tokenizer
import tokenizer_config as config
import light_cleaning


# Load the saved tokenizer (artifacts/tokenizer.json).
def load_tokenizer(path=config.TOKENIZER_PATH):
    return Tokenizer.from_file(str(path))


# Normalize the text like the training data, then turn it into token ids (with BOS / EOS by default).
def encode(tokenizer, text, add_bos_eos=True):
    return tokenizer.encode(light_cleaning.normalize(text), add_special_tokens=add_bos_eos).ids


# Turn token ids back into text (BOS / EOS / PAD are dropped).
def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True)


# Print a live example: the text, its tokens (as readable pieces), their ids, and the text decoded back.
def show_example(tokenizer, text):
    ids = encode(tokenizer, text)
    pieces = [tokenizer.decode([i], skip_special_tokens=False) for i in ids]
    decoded = decode(tokenizer, ids)
    print(f"  text:     {text}")
    print(f"  tokens:   {len(ids)} (with BOS / EOS), {len(text.split())} words")
    print(f"  pieces:   {' | '.join(pieces)}")
    print(f"  ids:      {ids}")
    print(f"  decoded:  {decoded}\n")


# Load the tokenizer and show a live example for every text.
def show_examples(texts):
    print("\n### 9) Encode a sentence into tokens, then decode it back\n")
    tokenizer = load_tokenizer()
    for text in texts:
        show_example(tokenizer, text)

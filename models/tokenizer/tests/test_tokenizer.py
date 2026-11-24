from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tokenizer_config as config
import train_tokenizer
import encode_text

# A tiny tokenizer trained on a few articles, enough to check the rules (the real one takes minutes).
ARTICLES = [
    "القاهرة\n\nمدينة عربية، وبها ١٢٣ شارع.\nسطر ثانٍ.",
    "Cairo\n\nAn English article with numbers 123 and punctuation!",
    "التشكيل محفوظ\n\nمُحَمَّد ذهب إلى المدرسة.",
]
TOKENIZER = train_tokenizer.train_on_texts(ARTICLES, vocab_size=512)

# Each case: (what it checks, text given to encode_text.encode, text expected after decode).
CASES = [
    ("[AR] Arabic text comes back exactly the same", ARTICLES[0], ARTICLES[0]),
    ("[AR] Tashkeel is kept", ARTICLES[2], ARTICLES[2]),
    ("[EN] English text comes back exactly the same", ARTICLES[1], ARTICLES[1]),
    ("[AR] New text: stretching tatweel is removed like in the training data", "محـــمد ذهب", "محمد ذهب"),
    ("[EN] New text: invisible characters are removed like in the training data", "Cairo​ is big", "Cairo is big"),
    ("[AR] New text: decomposed letters are joined (NFC) like in the training data", "أحمد", "أحمد"),
]


# Indent every line of a text so it reads clearly under its label.
def indent(text):
    return "\n".join(f"      {line}" for line in text.split("\n"))


# Run every case and the special-token checks, print them, and return the number of failures.
def run_cases():
    failures = 0
    for number, (name, text, expected) in enumerate(CASES, start=1):
        ids = encode_text.encode(TOKENIZER, text)
        after = encode_text.decode(TOKENIZER, ids)
        passed = after == expected
        failures += not passed
        print(f"{'✅' if passed else '❌'} {number}. {name}")
        print(f"   before:\n{indent(text)}")
        print(f"   after decode:\n{indent(after)}")
        if not passed:
            print(f"   expected:\n{indent(expected)}")
        print()

    tokens = TOKENIZER.encode(ARTICLES[0]).tokens
    checks = [
        ("every article starts with BOS and ends with EOS",
         tokens[0] == config.BOS_TOKEN and tokens[-1] == config.EOS_TOKEN),
        ("special tokens have ids 0, 1, 2, 3",
         [TOKENIZER.token_to_id(t) for t in config.SPECIAL_TOKENS] == [0, 1, 2, 3]),
        ("the vocabulary is not bigger than asked", TOKENIZER.get_vocab_size() <= 512),
    ]
    for name, passed in checks:
        failures += not passed
        print(f"{'✅' if passed else '❌'} {name}")

    total = len(CASES) + len(checks)
    print(f"\n{total - failures}/{total} passed")
    return failures


# Entry point for pytest: fails if any case fails.
def test_tokenizer_cases():
    assert run_cases() == 0


if __name__ == "__main__":
    sys.exit(1 if run_cases() else 0)

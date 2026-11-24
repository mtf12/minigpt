# Train the joint Arabic + English byte-level BPE tokenizer on the train split -> artifacts/tokenizer.json.
# The splits were already cleaned and normalized by the preprocessing, so the text goes in as is.
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
import pyarrow.parquet as pq

import tokenizer_config as config
import parquet_io


# Yield (article, language) from a split file, batch by batch. An article is "title\n\ntext".
def read_articles(path):
    for row in parquet_io.read_rows(path, columns=["title", "text", "language"]):
        yield f"{row['title'] or ''}\n\n{row['text'] or ''}", row["language"]


# Train a byte-level BPE tokenizer on a list of texts (`length` is only for the progress bar).
# Byte-level: text is split into UTF-8 bytes first, so no character is ever unknown.
def train_on_texts(texts, vocab_size=config.VOCAB_SIZE, length=None):
    tokenizer = Tokenizer(models.BPE(unk_token=config.UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=config.MIN_FREQUENCY,
        special_tokens=config.SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes are always in the vocabulary
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=length)

    # add BOS at the start and EOS at the end of every encoded text
    bos, eos = config.BOS_TOKEN, config.EOS_TOKEN
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A {eos}",
        special_tokens=[(bos, tokenizer.token_to_id(bos)), (eos, tokenizer.token_to_id(eos))],
    )
    return tokenizer


# Train on the train split and save to artifacts/tokenizer.json.
def train(vocab_size=config.VOCAB_SIZE, out_path=config.TOKENIZER_PATH):
    n = pq.ParquetFile(config.TRAIN_PATH).metadata.num_rows
    texts = (article for article, _ in read_articles(config.TRAIN_PATH))
    tokenizer = train_on_texts(texts, vocab_size, length=n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"vocabulary: {tokenizer.get_vocab_size():,}, saved to {out_path}")
    return tokenizer

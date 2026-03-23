from collections import Counter
import re

PAD_token = "<PAD>"
UNK_token = "<UNK>"


def normalize_text(text: str) -> str:

    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


def tokenize(text: str) -> list[str]:

    text = normalize_text(text)
    return text.split()


def build_vocab(texts: list[str], max_vocab_size: int = 25000) -> dict[str:int]:

    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    most_commom_tokens = counter.most_common(max_vocab_size - 2)

    vocab = {PAD_token: 0, UNK_token: 1}

    for idx, (token, _) in enumerate(most_commom_tokens, start=2):
        vocab[token] = idx

    return vocab


def numericalize(text: str, vocab: dict[str:int]) -> list[int]:

    tokens = tokenize(text)
    return [vocab.get(token, vocab[UNK_token]) for token in tokens]

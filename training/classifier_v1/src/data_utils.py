from collections import Counter

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .text_preprocessing import numericalize, PAD_token


def split_data(records, random_state: int = 10):

    texts = [r["texts"] for r in records]
    labels = [r["label"] for r in records]

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=random_state, stratify=labels
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    train_data = [{"text": x, "label": y} for x, y in zip(X_train, y_train)]
    val_data = [{"text": x, "label": y} for x, y in zip(X_val, y_val)]
    test_data = [{"text": x, "label": y} for x, y in zip(X_test, y_test)]

    return train_data, val_data, test_data


def truncate_or_pad(token_ids, max_length, pad_idx):

    token_ids = token_ids[:max_length]
    attention_length = len(token_ids)

    if attention_length < max_length:
        token_ids = token_ids + [pad_idx] * (max_length - attention_length)

    return token_ids, attention_length


class YelpReviewDataset(Dataset):

    def __init__(self, records, vocab, max_length):

        self.records = records
        self.vocab = vocab
        self.max_length = max_length
        self.pad_idx = vocab[PAD_token]

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        item = self.records[index]
        text = item["text"]
        label = item["label"]

        token_ids = numericalize(text, self.vocab)
        token_ids, seq_length = truncate_or_pad(
            token_ids, self.max_length, self.pad_idx
        )

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(seq_length, dtype=torch.long),
        }


def print_split_stats(name, records):

    labels = [r["label"] for r in records]
    count = Counter(labels)

    print(f"\n{name} size: {len(records)}")
    for label in sorted(count):
        print(f"label {label}: {count[label]}")

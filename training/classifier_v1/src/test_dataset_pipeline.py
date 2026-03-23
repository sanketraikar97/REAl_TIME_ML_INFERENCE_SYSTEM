from pathlib import Path

from .load_raw_data import load_reviews
from .data_utils import split_data, YelpReviewDataset, print_split_stats
from .text_preprocessing import build_vocab


def main():

    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / "data" / "yelp_dataset" / "yelp_academic_dataset_review.json"
    records = load_reviews(file_path)

    train_data, val_data, test_data = split_data(records)

    print_split_stats("Train", train_data)
    print_split_stats("Validation", val_data)
    print_split_stats("Test", test_data)

    train_data_texts = [r["text"] for r in train_data]
    vocab = build_vocab(train_data_texts)

    print(f"\nVocab size: {len(vocab)}")

    train_dataset = YelpReviewDataset(records=train_data, vocab=vocab, max_length=300)

    sample = train_dataset[0]

    print("\nSample item from dataset:")
    print("input_ids shape:", sample["input_ids"].shape)
    print("label:", sample["label"].item())
    print("length:", sample["length"].item())
    print("first 20 token ids:", sample["input_ids"][:20].tolist())


if __name__ == "__main__":
    main()

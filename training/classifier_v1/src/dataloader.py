from pathlib import Path
from torch.utils.data import DataLoader

from .load_raw_data import load_reviews
from .data_utils import YelpReviewDataset, split_data
from .text_preprocessing import build_vocab


def main():

    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / "data" / "yelp_dataset" / "yelp_academic_dataset_review.json"

    records = load_reviews(file_path)

    train_data, val_data, test_data = split_data(records)

    train_texts = [r["text"] for r in train_data]
    vocab = build_vocab(train_texts)

    train_dataset = YelpReviewDataset(train_data, vocab, max_length=300)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    batch = next(iter(train_loader))

    print("BATCH KEYS: ", batch.keys())
    print("INPUT ID SHAPE: ", batch["input_ids"].shape)
    print("LABEL SHAPE: ", batch["label"].shape)
    print("LENGTH SHAPE: ", batch["length"].shape)


if __name__ == "__main__":
    main()

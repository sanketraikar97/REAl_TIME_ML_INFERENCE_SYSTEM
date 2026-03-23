import torch

# from pathlib import Path
# from torch.utils.data import DataLoader

# from .load_raw_data import load_reviews
# from .data_utils import split_data, YelpReviewDataset
# from .text_preprocessing import build_vocab
# from .model_BiLSTM import BiLSTMClassifier


# def main():

#     base_dir = Path(__file__).resolve().parents[1]
#     file_path = base_dir / "data" / "yelp_dataset" / "yelp_academic_dataset_review.json"

#     records = load_reviews(file_path)
#     train_data, val_data, test_data = split_data(records)

#     train_texts = [r["text"] for r in train_data]
#     vocab = build_vocab(train_texts)

#     dataset = YelpReviewDataset(train_data, vocab, max_length=300)
#     loader = DataLoader(dataset, batch_size=32)

#     batch = next(iter(loader))

#     model = BiLSTMClassifier(vocab_size=len(vocab))

#     logits = model(input_ids=batch["input_ids"], lengths=batch["length"])

#     print("logits shape: ", logits.shape)


# if __name__ == "__main__":
#     main()
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

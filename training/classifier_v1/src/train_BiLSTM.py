from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .load_raw_data import load_reviews
from .data_utils import split_data, YelpReviewDataset
from .text_preprocessing import build_vocab
from .model_BiLSTM import BiLSTMClassifier


def train_epoch(model, dataloader, optimizer, criterion, device):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        length = batch["length"].to(device)
        label = batch["label"].to(device)

        logits = model(input_ids, length)
        loss = criterion(logits, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == label).sum().item()
        total_examples += label.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        length = batch["length"].to(device)
        label = batch["label"].to(device)

        logits = model(input_ids, length)
        loss = criterion(logits, label)

        total_loss += loss.item() * input_ids.size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == label).sum().item()
        total_examples += label.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / "data" / "yelp_dataset" / "yelp_academic_dataset_review.json"

    records = load_reviews(file_path, max_samples=200000)
    train_data, val_data, test_data = split_data(records)

    train_texts = [r["text"] for r in train_data]
    vocab = build_vocab(train_texts)

    train_dataset = YelpReviewDataset(train_data, vocab, max_length=300)
    val_dataset = YelpReviewDataset(val_data, vocab, max_length=300)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = BiLSTMClassifier(vocab_size=len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 100
    best_val_accuracy = 0
    early_stop = 3

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pt")

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.4f} | "
            f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f} | "
        )

        if val_accuracy < best_val_accuracy:
            early_stop -= 1
        else:
            early_stop = 3

        if early_stop == 0:
            print("best model has been acheived and early stop has been triggered")
            break


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from collections import Counter


def load_reviews(file_path: Path, max_samples: int | None = None):

    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):

            if max_samples is not None and i >= max_samples:
                break

            row = json.loads(line)

            texts = row.get("text", "").strip()
            stars = row.get("stars")

            if not texts:
                continue
            if stars not in {1, 2, 3, 4, 5}:
                continue

            label = stars - 1
            records.append({"texts": texts, "stars": stars, "label": label})
    return records


def summarize_data(records):

    print(f"Loaded records: {len(records)}")

    label_counts = Counter(r["label"] for r in records)
    print("\nLabel Distribbution:")

    for labels in sorted(label_counts):
        print(f"label: {labels} (stars= {labels + 1}): {label_counts[labels]}")

    lengths = [len(r["texts"].split()) for r in records]
    if lengths:
        print("\nApprox review lengths (in words):")
        print(f"min: {min(lengths)}")
        print(f"max: {max(lengths)}")
        print(f"avg: {sum(lengths) / len(lengths):.2f}")

    print("\nSample Records:")

    for idx, r in enumerate(records[:3], start=1):
        print(f"\nExample: {idx}")
        print(f"Stars: {r['stars']}")
        print(f"Label: {r['label']}")
        print(f"Text: {r['texts'][:]}")


def main():
    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / "data" / "yelp_dataset" / "yelp_academic_dataset_review.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path.resolve()}")

    records = load_reviews(file_path)  # max_samples=10000)
    summarize_data(records)


if __name__ == "__main__":
    main()

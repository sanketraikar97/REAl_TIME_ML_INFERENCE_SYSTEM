from pathlib import Path
from .load_raw_data import load_reviews
from .data_utils import split_data
from .text_preprocessing import build_vocab
from .embedding_utils import load_glove, build_embedding_matrix


def main():

    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / "data" / "yelp_dataset" / "yelp_academic_dataset_review.json"
    glove_path = base_dir / "data" / "glove6B" / "glove.6B.200d.txt"

    records = load_reviews(file_path, max_samples=200000)

    train_data, _, _ = split_data(records)

    train_texts = [r["text"] for r in train_data]
    vocab = build_vocab(train_texts)

    print("Loading GloVe...")
    glove_embeddings = load_glove(glove_path, embedding_dim=200)

    print(f"Loaded GloVe word vectors: {len(glove_embeddings)}")

    embedding_matrix, count, coverage = build_embedding_matrix(
        vocab, glove_embeddings, embedding_dim=200
    )

    print("Embedding matrix shape:", embedding_matrix.shape)
    print("Matched vocab words:", count)
    print(f"Coverage: {coverage:.2%}")


if __name__ == "__main__":
    main()

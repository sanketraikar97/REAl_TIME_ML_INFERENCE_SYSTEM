from pathlib import Path
import numpy as np


def load_glove(path: Path, embedding_dim: int = 200):

    embeddings = {}

    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = parts[1:]

            if len(vector) != embedding_dim:
                continue

            embed_vector = np.asarray(vector, dtype=np.float32)
            embeddings[word] = embed_vector

    return embeddings


def build_embedding_matrix(
    vocab: dict[str, int],
    glove_embeddings: dict[str, np.ndarray],
    embedding_dim: int = 200,
):

    vocab_size = len(vocab)

    embedding_matrix = np.random.normal(
        loc=0.0, scale=0.6, size=(vocab_size, embedding_dim)
    ).astype(np.float32)

    count = 0

    for token, idx in vocab.items():

        vector = glove_embeddings.get(token)
        if vector is not None:
            embedding_matrix[idx] = vector
            count += 1

    if "<PAD>" in vocab:
        embedding_matrix[vocab["<PAD>"]] = np.zeros(embedding_dim, dtype=np.float32)

    coverage = count / vocab_size

    return embedding_matrix, count, coverage

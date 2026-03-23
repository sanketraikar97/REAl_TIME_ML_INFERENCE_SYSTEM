from .text_preprocessing import tokenize, build_vocab, numericalize


def main():

    texts = [
        "This place was amazing and the food was great",
        "Terrible service and terrible food",
        "The food was okay but the service was slow",
    ]

    print("\ntokenizeed samples")
    print(tokenize(texts[0]))

    vocab = build_vocab(texts, max_vocab_size=25)

    for token, idx in vocab.items():
        print(f"{token}: {idx}")

    encoded_str = numericalize("Amazing food and slow service", vocab)
    print(encoded_str)


if __name__ == "__main__":
    main()

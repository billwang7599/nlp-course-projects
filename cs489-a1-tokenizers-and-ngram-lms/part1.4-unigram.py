import argparse

from unigram import check_lossless, train_unigram


def main() -> None:
    parser = argparse.ArgumentParser(description="CS489 A1 Part 1.4: train a unigram tokenizer with sentencepiece.")
    parser.add_argument("--input", required=True, help="cleaned directory, e.g. artifacts/part1/cleaned")
    parser.add_argument("--output", required=True, help="e.g. artifacts/part1/unigram/")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    args = parser.parse_args()

    model_path = train_unigram(args.input, args.output, args.vocab_size, args.character_coverage)
    print(f"wrote {model_path}")
    check_lossless(model_path, args.input)


if __name__ == "__main__":
    main()

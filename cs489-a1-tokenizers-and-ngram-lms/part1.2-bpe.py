import argparse

from bpe import BPETrainer
from data import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description="CS489 A1 Part 1.2: byte-level BPE training.")
    parser.add_argument("--input", required=True, help="cleaned directory, e.g. artifacts/part1/cleaned")
    parser.add_argument("--output", required=True, help="e.g. artifacts/part1/bpe/")
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args()
    text = load_split(args.input, "train")
    tokenizer = BPETrainer(args.vocab_size).train(text)
    tokenizer.save(args.output)
    print(f"wrote {len(tokenizer.vocab)} tokens and {len(tokenizer.merges)} merges to {args.output}")


if __name__ == "__main__":
    main()

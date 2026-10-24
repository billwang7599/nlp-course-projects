# Part 3: train one n-gram model on the cleaned train files and print its dev perplexity.
#
#   uv run python part3_ngram_lm.py --input artifacts/part1/cleaned --bpe artifacts/part1/bpe \
#       --order 3 --smoothing add-alpha --alpha 0.01
#
# --smoothing none is maximum likelihood (3.1); if a dev n-gram was never seen in
# training the perplexity is infinite and one such n-gram is printed.
#
import argparse

from bpe import BPETokenizer
from ngram import bits_per_char, count_ngrams, load_encoded, nll, perplexity


def main() -> None:
    parser = argparse.ArgumentParser(description="CS489 A1 Part 3: token-level n-gram language model.")
    parser.add_argument("--input", required=True, help="Part-1 cleaned directory")
    parser.add_argument("--bpe", required=True, help="Part-1 BPE folder, e.g. artifacts/part1/bpe")
    parser.add_argument("--order", type=int, required=True, help="n, the model order")
    parser.add_argument("--smoothing", choices=["none", "add-alpha"], default="none")
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()
    alpha = args.alpha if args.smoothing == "add-alpha" else 0.0

    tokenizer = BPETokenizer.load(args.bpe)
    train = load_encoded(args.input, "train", tokenizer)
    dev = load_encoded(args.input, "dev", tokenizer)

    counts = count_ngrams([line for lines, _ in train.values() for line in lines], args.order)
    dev_lines = [line for lines, _ in dev.values() for line in lines]
    dev_chars = sum(n_chars for _, n_chars in dev.values())
    dev_tokens = sum(len(line) for line in dev_lines)

    total_nll, missing = nll(counts, dev_lines, args.order, alpha)
    print(f"n={args.order} alpha={alpha}: {dev_tokens} dev tokens")
    print(f"dev perplexity: {perplexity(total_nll, dev_tokens)}")
    print(f"dev bits per character: {bits_per_char(total_nll, dev_chars)}")
    if missing is not None:
        print(f"probability 0: this n-gram never appears in training: {missing}")


if __name__ == "__main__":
    main()

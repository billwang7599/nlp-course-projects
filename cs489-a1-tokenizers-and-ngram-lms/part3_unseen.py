# Part 3.4 helper: for each order n, what share of the dev tokens have an n-gram that never
# appeared in training (so the maximum-likelihood probability is 0), and what share have a
# context that never appeared (so the model has no counts at all for that context).
#
#   uv run python part3_unseen.py --input artifacts/part1/cleaned --bpe artifacts/part1/bpe
#
import argparse

from bpe import BPETokenizer
from ngram import count_ngrams, load_encoded

ORDERS = [2, 3, 4, 5]


def main() -> None:
    parser = argparse.ArgumentParser(description="CS489 A1 Part 3.4: share of unseen n-grams in the dev set.")
    parser.add_argument("--input", required=True, help="Part-1 cleaned directory")
    parser.add_argument("--bpe", required=True, help="Part-1 BPE folder, e.g. artifacts/part1/bpe")
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.bpe)
    train = load_encoded(args.input, "train", tokenizer)
    dev = load_encoded(args.input, "dev", tokenizer)
    counts = count_ngrams([line for lines, _ in train.values() for line in lines], max(ORDERS))
    dev_lines = [line for lines, _ in dev.values() for line in lines]

    for n in ORDERS:
        tokens = unseen_ngram = unseen_context = 0
        for line in dev_lines:
            for t, token in enumerate(line):
                m = min(t, n - 1)
                context = tuple(line[t - m:t])
                tokens += 1
                unseen_ngram += counts[m + 1].get(context + (token,), 0) == 0
                unseen_context += m >= 1 and counts[m].get(context, 0) == 0
        print(f"n={n}: {100 * unseen_ngram / tokens:.1f}% of dev tokens have an unseen n-gram, "
              f"{100 * unseen_context / tokens:.1f}% have an unseen context", flush=True)


if __name__ == "__main__":
    main()

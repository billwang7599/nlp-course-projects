# Part 3: try n = 2..5 and several alphas, pick the best by dev perplexity, save it.
#
#   uv run python part3.py --input artifacts/part1/cleaned --bpe artifacts/part1/bpe --output artifacts/part3/
#
# The model is trained once at order 5; lower orders reuse the same count tables.
#
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt

from bpe import BPETokenizer
from ngram import bits_per_char, count_ngrams, load_encoded, nll, perplexity, to_model
from part3_artifacts import save_ngram_model

ORDERS = [2, 3, 4, 5]
ALPHAS = [0.0, 1.0, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0001, 0.00001, 0.000001]  # 0.0 is plain maximum likelihood


# Scores every dev language for one (n, alpha). Returns one row per language plus an "all" row
# (the NLL, tokens and characters of the languages added together).
def score_all(counts, dev, n: int, alpha: float) -> list[dict]:
    rows, sum_nll, sum_tokens, sum_chars = [], 0.0, 0, 0
    for language, (lines, n_chars) in dev.items():
        total_nll, missing = nll(counts, lines, n, alpha)
        n_tokens = sum(len(line) for line in lines)
        rows.append({"n": n, "alpha": alpha, "language": language, "perplexity": perplexity(total_nll, n_tokens),
                     "bpc": bits_per_char(total_nll, n_chars), "missing": missing})
        sum_nll, sum_tokens, sum_chars = sum_nll + total_nll, sum_tokens + n_tokens, sum_chars + n_chars
    rows.append({"n": n, "alpha": alpha, "language": "all", "perplexity": perplexity(sum_nll, sum_tokens),
                 "bpc": bits_per_char(sum_nll, sum_chars), "missing": None})
    return rows


def plot(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    languages = sorted({r["language"] for r in rows} - {"all"})
    for language in languages:
        ys = [min(r["perplexity"] for r in rows if r["language"] == language and r["n"] == n) for n in ORDERS]
        ax.plot(ORDERS, ys, "-o", label=language, markersize=4)
    ax.set_yscale("log")
    ax.set_xticks(ORDERS)
    ax.set_xlabel("order n")
    ax.set_ylabel("dev perplexity (log scale)")
    ax.set_title("Perplexity against n (best alpha for each language and n)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "perplexity_vs_n.png", dpi=200)
    fig.savefig(output / "perplexity_vs_n.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CS489 A1 Part 3: pick the best n-gram model and save it.")
    parser.add_argument("--input", required=True, help="Part-1 cleaned directory")
    parser.add_argument("--bpe", required=True, help="Part-1 BPE folder, e.g. artifacts/part1/bpe")
    parser.add_argument("--output", required=True, help="e.g. artifacts/part3/")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer.load(args.bpe)
    train = load_encoded(args.input, "train", tokenizer)
    dev = load_encoded(args.input, "dev", tokenizer)
    counts = count_ngrams([line for lines, _ in train.values() for line in lines], max(ORDERS))
    print("counted n-grams:", {k: len(table) for k, table in counts.items()}, flush=True)

    rows = []
    for n in ORDERS:
        for alpha in ALPHAS:
            new_rows = score_all(counts, dev, n, alpha)
            rows += new_rows
            overall = new_rows[-1]
            print(f"n={n} alpha={alpha}: perplexity {overall['perplexity']:.2f}, bpc {overall['bpc']:.4f}", flush=True)
            if alpha == 0.0:
                missing = next((r["missing"] for r in new_rows if r["missing"] is not None), None)
                if missing is not None:
                    print(f"  maximum likelihood is infinite; unseen n-gram: {missing}", flush=True)

    with open(output / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "alpha", "language", "perplexity", "bpc"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    plot(rows, output)
    finite = [r for r in rows if r["language"] == "all" and math.isfinite(r["perplexity"])]

    best = min(finite, key=lambda r: r["perplexity"])
    print(f"best: n={best['n']} alpha={best['alpha']} perplexity {best['perplexity']:.2f} bpc {best['bpc']:.4f}")
    model_path = output / "model.json.gz"
    save_ngram_model(to_model(counts, best["n"], best["alpha"]), model_path)
    print(f"wrote {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

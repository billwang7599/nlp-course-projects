import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import sentencepiece as spm

from bpe import BPETrainer
from data import load_split
from unigram import train_unigram

VOCAB_SIZES = [5000, 10000, 20000, 50000]
# With character_coverage 0.9995 about 5,000 characters need their own piece,
# which does not fit in a vocabulary of 5,000, so that size uses 0.999.
CHARACTER_COVERAGE = {5000: 0.999}
DEFAULT_COVERAGE = 0.9995

TYPOLOGY = {
    "Agglutinative": ["fi", "hu"],
    "Fusional": ["de", "ru", "ar"],
    "Isolating": ["zh", "vi"],
}
NAMES = {"fi": "Finnish", "hu": "Hungarian", "de": "German", "ru": "Russian", "ar": "Arabic", "zh": "Mandarin", "vi": "Vietnamese"}
COLORS = {"fi": "tab:blue", "hu": "tab:cyan", "de": "tab:orange", "ru": "tab:red", "ar": "tab:brown", "zh": "tab:green", "vi": "tab:olive"}


# Tokens per UTF-8 byte of the cleaned dev file of each language (lower is better).
def measure(encode, dev_texts: dict[str, str]) -> dict[str, tuple[int, int]]:
    return {lang: (len(encode(text)), len(text.encode("utf-8"))) for lang, text in dev_texts.items()}


def save_results(rows: list[dict], output: Path) -> None:
    with open(output / "compression.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tokenizer", "vocab_size", "language", "tokens", "bytes", "ratio"])
        writer.writeheader()
        writer.writerows(rows)


# One panel per typology, one color per language: solid line = BPE, dashed = unigram.
def plot(rows: list[dict], output: Path) -> None:
    ratio = {(r["tokenizer"], r["vocab_size"], r["language"]): r["ratio"] for r in rows}
    fig, axes = plt.subplots(1, len(TYPOLOGY), figsize=(13, 4), sharey=True)
    for ax, (group, languages) in zip(axes, TYPOLOGY.items()):
        for lang in languages:
            for tokenizer, style in (("BPE", "-o"), ("unigram", "--s")):
                ys = [ratio[(tokenizer, v, lang)] for v in VOCAB_SIZES]
                ax.plot(VOCAB_SIZES, ys, style, color=COLORS[lang], label=f"{NAMES[lang]} {tokenizer}", markersize=4)
        ax.set_xscale("log")
        ax.set_xticks(VOCAB_SIZES)
        ax.set_xticklabels([f"{v // 1000}k" for v in VOCAB_SIZES])
        ax.minorticks_off()
        ax.set_title(group)
        ax.set_xlabel("vocabulary size V")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("compression ratio (tokens per byte, lower is better)")
    fig.suptitle("Compression ratio on the cleaned dev files")
    fig.tight_layout()
    fig.savefig(output / "compression_vs_vocab.png", dpi=200)
    fig.savefig(output / "compression_vs_vocab.pdf")
    plt.close(fig)


def print_table(rows: list[dict]) -> None:
    ratio = {(r["tokenizer"], r["vocab_size"], r["language"]): r["ratio"] for r in rows}
    print(f"{'':10s}" + "".join(f"{t + ' ' + str(v // 1000) + 'k':>14s}" for t in ("BPE", "unigram") for v in VOCAB_SIZES))
    for lang in NAMES:
        print(f"{NAMES[lang]:10s}" + "".join(f"{ratio[(t, v, lang)]:>14.4f}" for t in ("BPE", "unigram") for v in VOCAB_SIZES))


def main() -> None:
    parser = argparse.ArgumentParser(description="CS489 A1 Part 1.5: compression ratio against vocabulary size.")
    parser.add_argument("--input", required=True, help="cleaned directory, e.g. artifacts/part1/cleaned")
    parser.add_argument("--output", required=True, help="e.g. artifacts/part1/analysis/")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_text = load_split(args.input, "train")
    dev_texts = {lang: (Path(args.input) / lang / "dev.txt").read_text(encoding="utf-8") for lang in NAMES}

    rows = []
    for vocab_size in VOCAB_SIZES:
        print(f"=== V = {vocab_size}: BPE", flush=True)
        bpe = BPETrainer(vocab_size).train(train_text)
        results = {"BPE": measure(bpe.encode, dev_texts)}

        print(f"=== V = {vocab_size}: unigram", flush=True)
        coverage = CHARACTER_COVERAGE.get(vocab_size, DEFAULT_COVERAGE)
        model_path = train_unigram(args.input, str(output / f"unigram_{vocab_size}"), vocab_size, coverage)
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        results["unigram"] = measure(sp.encode, dev_texts)

        for tokenizer, per_language in results.items():
            for lang, (tokens, n_bytes) in per_language.items():
                rows.append({"tokenizer": tokenizer, "vocab_size": vocab_size, "language": lang,
                             "tokens": tokens, "bytes": n_bytes, "ratio": round(tokens / n_bytes, 6)})
        save_results(rows, output)  # keep partial results if a later size fails

    plot(rows, output)
    print_table(rows)
    print(f"wrote {output / 'compression.csv'} and {output / 'compression_vs_vocab.png'}")


if __name__ == "__main__":
    main()

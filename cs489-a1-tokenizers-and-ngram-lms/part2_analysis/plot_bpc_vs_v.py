# Plots the Part-2 vocabulary sweep: dev bpc against V, and the Part-1 compression
# (tokens per byte) against V for the same tokenizers.
#
#   uv run python part2_analysis/plot_bpc_vs_v.py
#
# sweep_results.csv holds one row per V. The "source" column says where the bpc came from:
# "fixed evaluator" = part2_eval_bpc.py (artifacts/sweep/v*/eval.txt),
# "training log" = the final dev bpc printed by part2.py on Colab (the checkpoints
# for these two sizes were lost, so the fixed evaluator could not be run on them).
#
# AI-assisted (Claude): this script was written with Claude's help.

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt

HERE = Path(__file__).parent


def main() -> None:
    with open(HERE / "sweep_results.csv", newline="", encoding="utf-8") as f:
        rows = [{**r, "vocab_size": int(r["vocab_size"]), "dev_bpc": float(r["dev_bpc"]),
                 "tokens_per_byte": float(r["tokens_per_byte"])} for r in csv.DictReader(f)]
    sizes = [r["vocab_size"] for r in rows]
    best = min(rows, key=lambda r: r["dev_bpc"])

    fig, (ax_bpc, ax_ratio) = plt.subplots(1, 2, figsize=(11, 4))

    ax_bpc.plot(sizes, [r["dev_bpc"] for r in rows], "-", color="tab:blue")
    for source, marker_fill in (("fixed evaluator", "tab:blue"), ("training log", "white")):
        points = [r for r in rows if r["source"] == source]
        ax_bpc.plot([r["vocab_size"] for r in points], [r["dev_bpc"] for r in points], "o",
                    markerfacecolor=marker_fill, color="tab:blue", label=source)
    ax_bpc.annotate(f"best: V={best['vocab_size']}, {best['dev_bpc']:.3f}", (best["vocab_size"], best["dev_bpc"]),
                    textcoords="offset points", xytext=(10, -22), arrowprops={"arrowstyle": "->"})
    ax_bpc.set_ylabel("dev bits per character (lower is better)")
    ax_bpc.set_title("bpc against vocabulary size")
    ax_bpc.legend()

    ax_ratio.plot(sizes, [r["tokens_per_byte"] for r in rows], "-o", color="tab:orange")
    ax_ratio.set_ylabel("compression ratio on dev (tokens per byte, lower is better)")
    ax_ratio.set_title("Part-1 compression for the same tokenizers")

    for ax in (ax_bpc, ax_ratio):
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{v / 1000:g}k" for v in sizes])
        ax.minorticks_off()
        ax.set_xlabel("vocabulary size V")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(HERE / "bpc_vs_vocab.png", dpi=200)
    fig.savefig(HERE / "bpc_vs_vocab.pdf")
    print(f"wrote {HERE / 'bpc_vs_vocab.png'}")


if __name__ == "__main__":
    main()

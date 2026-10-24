# Part 2: train a tokenizer, encode the cleaned text, train the fixed Transformer.
#
#   uv run python part2.py --input artifacts/part1/cleaned --output artifacts/part2/ --vocab-size 50000
#
# --prepare-only trains and encodes only (no GPU needed). --skip-prepare runs the
# model training from files already in --output, so encoding can be done on a laptop.
import argparse
import json
from pathlib import Path

import torch

from data import load_split, write_ids
from part2_tokenizer import load_tokenizer
from part2_train import train


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Trains the tokenizer and saves it to {output}/tokenizer/. Returns its settings.
def train_tokenizer(args: argparse.Namespace, tokenizer_dir: Path) -> dict:
    if args.tokenizer == "bpe":
        from bpe import BPETrainer

        barriers = {"\n"} if args.newline_barrier else set()
        tokenizer = BPETrainer(args.vocab_size, barriers=barriers).train(load_split(args.input, "train"))
        tokenizer.save(tokenizer_dir)
        return {"vocab_size": args.vocab_size, "barriers": sorted(barriers), "source": tokenizer_dir.name + "/"}
    from unigram import train_unigram

    coverage = 0.999 if args.vocab_size <= 5000 else 0.9995  # 0.9995 needs about 5,000 pieces for characters
    train_unigram(args.input, str(tokenizer_dir), args.vocab_size, coverage)
    return {"vocab_size": args.vocab_size, "character_coverage": coverage, "split_by_whitespace": False,
            "source": tokenizer_dir.name + "/"}


def prepare(args: argparse.Namespace, output: Path) -> None:
    tokenizer_dir = output / "tokenizer"
    settings = train_tokenizer(args, tokenizer_dir)
    (output / "tokenizer_settings.json").write_text(
        json.dumps({"algorithm": args.tokenizer, "settings": settings}, indent=2) + "\n", encoding="utf-8"
    )

    tokenizer = load_tokenizer(str(tokenizer_dir))
    dev_text = load_split(args.input, "dev")
    with open(output / "dev_text.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(dev_text)
    for split, text in (("train", load_split(args.input, "train")), ("dev", dev_text)):
        ids = tokenizer.encode(text)
        assert max(ids) < args.vocab_size, "the tokenizer made an id outside the vocabulary"
        write_ids(output / f"{split}.bin", ids)
        print(f"{split}: {len(text.encode('utf-8'))} bytes -> {len(ids)} tokens (ratio {len(ids) / len(text.encode('utf-8')):.4f})", flush=True)


def train_model(args: argparse.Namespace, output: Path) -> None:
    device = pick_device()
    print(f"training on {device}", flush=True)
    metrics = train(output / "train.bin", output / "dev.bin", args.vocab_size, output,
                    dev_text_path=output / "dev_text.txt", device=device)

    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    config["tokenizer"] = json.loads((output / "tokenizer_settings.json").read_text(encoding="utf-8"))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    dev_bytes = len((output / "dev_text.txt").read_bytes())
    metrics.update(vocab_size=args.vocab_size, tokenizer=config["tokenizer"]["algorithm"],
                   dev_bytes=dev_bytes, dev_tokens_per_byte=(metrics["dev_tokens_scored"] + 1) / dev_bytes)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CS489 A1 Part 2: tokenizer + fixed Transformer.")
    parser.add_argument("--input", required=True, help="Part-1 cleaned directory")
    parser.add_argument("--output", required=True, help="e.g. artifacts/part2/")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--tokenizer", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("--newline-barrier", action="store_true", help="bpe only: merges may not touch a newline")
    parser.add_argument("--prepare-only", action="store_true", help="train and encode, but do not train the model")
    parser.add_argument("--skip-prepare", action="store_true", help="train the model from files already in --output")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_prepare:
        prepare(args, output)
    if not args.prepare_only:
        train_model(args, output)


if __name__ == "__main__":
    main()

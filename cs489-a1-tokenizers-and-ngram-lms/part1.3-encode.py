import argparse
from pathlib import Path

from bpe import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="CS489 A1 Part 1.3: encode a text file into token ids.")
    parser.add_argument("--input", required=True, help="text file to encode")
    parser.add_argument("--bpe", required=True, help="directory with bpe_vocab.txt and bpe_merges.txt")
    parser.add_argument("--output", required=True, help="file to write: one token id per line")
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.bpe)
    with open(args.input, encoding="utf-8", newline="") as f:  # newline="": no \r\n translation
        text = f.read()
    ids = tokenizer.encode(text)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{token_id}\n" for token_id in ids), encoding="ascii")
    print(f"{args.input}: {len(text.encode('utf-8'))} bytes -> {len(ids)} tokens")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from bpe import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="CS489 A1 Part 1.3: decode token ids back into the original bytes.")
    parser.add_argument("--input", required=True, help="file with one token id per line")
    parser.add_argument("--bpe", required=True, help="directory with bpe_vocab.txt and bpe_merges.txt")
    parser.add_argument("--output", required=True, help="file to write: the exact original bytes")
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.bpe)
    ids = [int(line) for line in Path(args.input).read_text(encoding="ascii").split()]
    data = tokenizer.decode(ids)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)  # bytes, so nothing is translated or re-encoded
    print(f"{args.input}: {len(ids)} tokens -> {len(data)} bytes")


if __name__ == "__main__":
    main()

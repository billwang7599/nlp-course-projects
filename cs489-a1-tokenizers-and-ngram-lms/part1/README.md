# A1 Part 1: Cleaning and tokenizers

Wikipedia dumps for seven languages are cleaned, then tokenized with byte-level BPE (written from scratch) and a
unigram tokenizer (SentencePiece).

## Files (all in the project root)

- `part1.1-clean.py`, `part1.1-stats.py`: a pipeline of small steps that strips wiki markup (HTML, templates, tables,
  links, headings, lists, entities, invisible characters), and a script that reports raw vs. cleaned byte counts
- `bpe.py`: `BPETrainer`, `BPETokenizer` (byte-level BPE, any text encodes and decodes losslessly)
- `part1.2-bpe.py`, `part1.3-encode.py`, `part1.3-decode.py`: train, encode and decode
- `unigram.py`, `part1.4-unigram.py`: unigram tokenizer with a lossless round-trip check
- `part1.5-analysis.py`: compression ratio vs. vocabulary size for both tokenizers

## Run (from the project root)

```bash
uv run python part1.1-clean.py --input data/raw --output artifacts/part1/cleaned
uv run python part1.1-stats.py --raw-input data/raw --clean-input artifacts/part1/cleaned
uv run python part1.2-bpe.py --input artifacts/part1/cleaned --output artifacts/part1/bpe --vocab-size 50000
uv run python part1.3-encode.py --input artifacts/part1/cleaned/de/dev.txt --bpe artifacts/part1/bpe --output artifacts/part1/encoded/de_dev.ids
uv run python part1.3-decode.py --input artifacts/part1/encoded/de_dev.ids --bpe artifacts/part1/bpe --output artifacts/part1/decoded/de_dev.txt
uv run python part1.4-unigram.py --input artifacts/part1/cleaned --output artifacts/part1/unigram --vocab-size 50000
uv run python part1.5-analysis.py --input artifacts/part1/cleaned --output artifacts/part1/analysis
```

BPE training at V = 50,000 takes about 3 minutes. `part1.5-analysis.py` trains 8 tokenizers and is slow.

## Results

- Cleaning removes roughly 40 to 55% of the raw bytes (Mandarin: 54.6M raw to 24.4M cleaned training bytes).
- Compression ratio (tokens per byte) of BPE at V = 50,000 ranges from 0.156 (Russian) to 0.260 (German). From
  V = 5,000 to 50,000 it improves 36 to 42% for every language.
- BPE beats unigram at most vocabulary sizes, with the biggest gap at 5k.
- The write-up splits compression into bytes per character and tokens per character, and relates it to each
  language's morphology. See `writeup.md`, section 1.5.

![Compression ratio against vocabulary size](../figures/compression_vs_vocab.png)

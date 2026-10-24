# CS 489 A1: Tokenizers and n-gram language models

> Coursework for CS 489 (University of Waterloo, Fall 2026). Students in this or a similar course:
> please do not read or copy this, as it may violate academic integrity rules.

Data cleaning, tokenizers, and statistical language models for seven languages (Finnish, Hungarian, German,
Russian, Arabic, Mandarin, Vietnamese) of Wikipedia text.

| Part | What | README |
|---|---|---|
| 1 | Clean the raw dumps, byte-level BPE, encoder/decoder, unigram tokenizer, compression vs. vocabulary size | [part1](part1/README.md) |
| 2 | Transformer bits-per-character across vocabulary sizes | [part2](part2/README.md) |
| 3 | n-gram language model with add-alpha smoothing | [part3](part3/README.md) |

The full analysis is in [`writeup.md`](writeup.md).

## Setup

```bash
uv sync
uvx --from huggingface_hub hf download compling/cs489-f26-a1 --repo-type dataset --local-dir data
```

Run every command from this folder (the part READMEs list them). The code is flat so the scripts can import each
other (`bpe.py` is shared by parts 1 to 3).

## Course-provided files (not included)

`data.py`, `part2_transformer_lm.py`, `part2_train.py`, `part2_eval_bpc.py` and `part3_artifacts.py` are the
unmodified starter files from the course, so they are not in this repo. Put them in this folder to run parts 2 and 3.

## AI assistance

Claude Code helped me learn the material and write much of the code. I wrote the `TextCleaner` class and pipeline
idea, `encode` and `decode` in `bpe.py`, and the first version of `count_ngrams` myself, and Claude reviewed and
fixed bugs in them. I wrote my analysis in my own words, and Claude helped draft it into clearer text, as the
write-up declares.

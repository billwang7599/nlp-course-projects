# NLP Course Projects

My assignments from **CS 489: Introduction to Natural Language Processing** (University of Waterloo, Fall 2026).

> **Notice to students:** this is coursework. If you are taking CS 489 or a similar course, please **do not read,
> copy, or adapt this code or the write-ups** for your own assignments. Doing so may violate your
> university's academic integrity policy. Come back after your course ends.

| Project | Topic | Highlights |
|---|---|---|
| [A1: Tokenizers and n-gram LMs](cs489-a1-tokenizers-and-ngram-lms/) | Text cleaning, byte-level BPE and unigram tokenizers, tokenizer-vs-compression study, n-gram language models | Byte-level BPE from scratch, bits-per-character study across 8 vocabulary sizes, add-alpha n-gram LM |
| [A2: LSTM and Transformer LMs](cs489-a2-lstm-and-transformer-lms/) | Byte-level neural language models in PyTorch | LSTM and decoder-only Transformer, YAML-driven training, W&B logging, hyperparameter sweeps |

Each project has its own README, and a README for each part with its files, how to run it, and results.

## AI assistance

I used Claude Code (Anthropic) on both assignments, as declared in each write-up: to explain concepts and libraries,
and to help write code. The per-project READMEs say what I wrote myself and what Claude wrote.

## Not included

Assignment handouts, course-provided starter files (such as `evaluator.py`), datasets, checkpoints and logs.
Each project README says which course files you would need to run it.

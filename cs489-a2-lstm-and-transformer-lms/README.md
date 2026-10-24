# CS 489 A2: LSTM and Transformer language models

> Coursework for CS 489 (University of Waterloo, Fall 2026). Students in this or a similar course:
> please do not read or copy this, as it may violate academic integrity rules.

Byte-level language models trained from scratch on a Wikipedia corpus (`compling/cs489-f26-a2`): vocabulary of 258
(256 bytes, BOS, PAD), fixed blocks of at most 256 bytes. Models are compared by dev perplexity.

| Part | Model | README |
|---|---|---|
| 1 | LSTM | [part1](part1/README.md) |
| 2 | Small decoder-only Transformer | [part2](part2/README.md) |

Everything is configured from YAML (`configs/`), and runs log to Weights & Biases (offline by default) and to CSV.

## Setup and run

```bash
uv sync
uv run python run_part1.py --config configs/lstm.yaml          # LSTM
uv run python run_part2.py --config configs/transformer.yaml   # Transformer
```

Training writes the best-dev checkpoint, a config copy, a metrics CSV and a loss plot next to the config
(`../artifacts/partN`, `../results`). Load a checkpoint and print its dev perplexity:

```bash
uv run python submission.py --config artifacts/part1/config.yaml
```

To log to W&B online, put your own key in your shell only: `export WANDB_API_KEY=...` and `export WANDB_MODE=online`.

## Layout

- `run_part1.py`, `run_part2.py`: training, per-epoch dev evaluation, checkpointing, logging
- `model.py`: LSTM language model and loss. `transformer.py`: Transformer language model
- `data.py`: byte blocks, BOS/PAD handling, batching
- `submission.py`: rebuilds a model from its config, loads the checkpoint, evaluates
- `sweep.py`: hyperparameter grids (size and learning rate, then depth against epochs)
- `chat.py`: a terminal REPL that continues text with a trained model (temperature and top-k sampling)
- `configs/`: the YAML for each model

## Course-provided file (not included)

`evaluator.py` (the dev-perplexity evaluator and the 258-token constants) is provided by the course. Put it in this
folder to run the code.

## AI assistance

Claude Code explained PyTorch concepts and APIs. I wrote the LSTM (`model.py`) myself and designed the Transformer
(token plus learned position embeddings, a causal self-attention stack, a linear output layer). Claude wrote the
Transformer module to match, and wrote the data pipeline, training scripts, loader, sweep script and this README.

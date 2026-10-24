# A2 Part 1: LSTM language model

A byte-level LSTM language model, written with `nn.LSTM`.

## Model

`nn.Embedding` (256-dim, PAD ignored) then dropout, a 2-layer `nn.LSTM` (hidden size 768, dropout between layers),
dropout, and a linear layer to the 258-token vocabulary. About 8.14M parameters.

## Training

Adam (learning rate 2e-3), gradient clipping at 1.0, batch size 64, 8 epochs, seed 489, float32. Blocks are shuffled
every epoch. After each epoch the full dev set is scored by the course evaluator, and only the best-dev checkpoint
is kept.

## Files (project root)

`model.py`, `run_part1.py`, `data.py`, `submission.py`, `configs/lstm.yaml`

## Run (from the project root)

```bash
uv run python run_part1.py --config configs/lstm.yaml
uv run python submission.py --config artifacts/part1/config.yaml
```

## Tuning

I swept hidden size (512, 768) and learning rate (1e-3, 2e-3) on a Colab T4 at 8 epochs. The best, hidden 768 with
learning rate 2e-3, reached dev perplexity **3.603** in about 30.5 minutes. A larger learning rate and a larger
model each helped. A second sweep trades layers against epochs under a 35-minute budget.

## Results

Final dev perplexity from the submitted run: **TBD** (the sweep result above is from a tuning run).

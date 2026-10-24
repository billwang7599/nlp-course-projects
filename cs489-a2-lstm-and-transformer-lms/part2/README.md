# A2 Part 2: Transformer language model

A small decoder-only Transformer over the same byte vocabulary, built from PyTorch's `nn.TransformerEncoder` blocks.
Despite PyTorch's "encoder" name, the causal attention mask makes it decoder-only.

## Model

Token embedding plus a learned position embedding (summed, then dropout), 4 pre-norm blocks (4 heads, d_model 384,
feed-forward 1536), a final LayerNorm, and a linear layer to the 258-token vocabulary. About 7.40M parameters.

## Training

Adam (learning rate 1e-3), gradient clipping at 1.0, batch size 32, 8 epochs, seed 489, float32. The loop is the
same as Part 1, so `run_part2.py` differs only in how it builds the model.

## Files (project root)

`transformer.py`, `run_part2.py`, `data.py`, `submission.py`, `configs/transformer.yaml`

## Run (from the project root)

```bash
uv run python run_part2.py --config configs/transformer.yaml
uv run python submission.py --config artifacts/part2/config.yaml
```

## Tuning

I swept d_model (256, 384) and learning rate (3e-4, 1e-3) on a Colab T4 at 8 epochs. The best, d_model 384 with
learning rate 1e-3, reached dev perplexity **3.610** in about 27.4 minutes. A larger learning rate and a larger
model each helped. A second sweep trades layers against epochs under a 35-minute budget.

## Results

Final dev perplexity from the submitted run: **TBD** (the sweep result above is from a tuning run).

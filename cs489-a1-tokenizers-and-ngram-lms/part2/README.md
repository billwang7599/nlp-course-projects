# A1 Part 2: Bits per character

A fixed Transformer language model (course-provided) is trained on tokens from my byte-level BPE, and compared
by **bits per character (bpc)**, so tokenizers with different vocabulary sizes can be compared fairly.

## Files (project root unless noted)

- `part2.py`: trains the tokenizer, encodes the data, trains the Transformer, writes the artifacts
  (`--prepare-only` for tokenizer and encoding without a GPU, `--skip-prepare` to train from existing files)
- `part2_tokenizer.py`: tokenizer loading
- `part2_analysis/plot_bpc_vs_v.py`, `part2_analysis/sweep_results.csv`: the bpc vs. vocabulary-size plot and its data
- Course-provided and not included: `part2_transformer_lm.py`, `part2_train.py`, `part2_eval_bpc.py`

## Run (from the project root; needs a GPU, I used a Colab T4)

```bash
uv run python part2.py --input artifacts/part1/cleaned --output artifacts/part2/ --vocab-size 2000
uv run python part2_eval_bpc.py --config artifacts/part2/config.json --model artifacts/part2/model.pt \
    --ids artifacts/part2/dev.bin --text artifacts/part2/dev_text.txt
```

The vocabulary sweep runs the same command with `--vocab-size` set to 500, 1000, 2000, 3000, 5000, 10000, 20000
and 50000, each into its own folder. Then: `uv run python part2_analysis/plot_bpc_vs_v.py`.

## Results

- Baseline (V = 50,000): dev bpc **3.161**, token perplexity 975.
- Best (V = 2,000): dev bpc **2.959**.
- The 500 and 1,000 values come from the training log, not the fixed evaluator (I ran out of GPU credits before
  saving those checkpoints). The write-up says so, section 2.3.

![Dev bpc and compression ratio against vocabulary size](../figures/bpc_vs_vocab.png)

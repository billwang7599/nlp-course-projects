"""Part-2 tokenizer interface.

The TAs score your Part-2 checkpoint roughly as follows:

    import part2_tokenizer
    tokenizer = part2_tokenizer.load_tokenizer("artifacts/part2/tokenizer/")
    ids = tokenizer.encode(test_text)

Contract:
- load_tokenizer(path) reloads everything it needs from the given directory
  (your submitted artifacts/part2/tokenizer/ folder) and returns a tokenizer
  object.
- tokenizer.encode(text) maps a cleaned text string to a list of token ids,
  each in {0, ..., V - 1} for your submitted vocabulary size V.
- Encoding must be lossless because bits-per-character divides by the full
  character count of the text. Decoding the resulting token IDs must
  reproduce the input text exactly, including line breaks, repeated
  whitespace, and unseen characters: decode(encode(text)) == text.
"""

from __future__ import annotations

from pathlib import Path


# Wraps a sentencepiece model so that it has the same encode(text) as BPETokenizer.
class SentencePieceTokenizer:
    def __init__(self, model_file: Path) -> None:
        import sentencepiece as spm

        self._sp = spm.SentencePieceProcessor(model_file=str(model_file))

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text)


# The folder holds either unigram.model (sentencepiece) or bpe_vocab.txt and
# bpe_merges.txt (my byte-level BPE, see bpe.py).
def load_tokenizer(path: str):
    directory = Path(path)
    if (directory / "unigram.model").is_file():
        return SentencePieceTokenizer(directory / "unigram.model")
    if (directory / "bpe_vocab.txt").is_file():
        from bpe import BPETokenizer

        return BPETokenizer.load(directory)
    raise FileNotFoundError(f"no tokenizer files (unigram.model or bpe_vocab.txt) in {directory}")

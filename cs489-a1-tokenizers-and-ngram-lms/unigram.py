import os
from pathlib import Path

import sentencepiece as spm

from data import iter_split_files

# Strings that sentencepiece's default settings do not reproduce exactly.
LOSSLESS_TESTS = [
    "line one\nline two\n",
    "a\n\n\nb",
    "a  b   c",
    "  leading and trailing  ",
    "a\tb",
    "a\r\nb",
    "Ａﬁ①",  # fullwidth A, fi ligature, circled 1
    "é",  # e + combining accent
    "\U0001f600 emoji",
    "褘髓",  # Chinese characters
    "Hello, world.",
]


# Trains a unigram model on the cleaned train files of every language and
# writes {output_dir}/unigram.model (and unigram.vocab).
# The settings turn off everything sentencepiece normally rewrites, so that
# decode(encode(text)) == text:
#   identity normalization: no Unicode rewriting (it would change e.g. ligatures)
#   remove_extra_whitespaces=False, add_dummy_prefix=False: keep spaces exactly
#   byte_fallback=True: a character with no piece becomes its bytes, not <unk>
# and for better compression:
#   split_by_whitespace=False: a piece may contain a space
#   character_coverage: the share of the training text whose characters all get
#   a piece of their own; the rarest characters use byte fallback instead
def train_unigram(input_dir: str, output_dir: str, vocab_size: int, character_coverage: float = 0.9995) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_files = [str(path) for path in iter_split_files(input_dir, "train")]  # sorted, like load_split
    spm.SentencePieceTrainer.train(
        input=",".join(train_files),
        model_prefix=str(output / "unigram"),
        model_type="unigram",
        vocab_size=vocab_size,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        byte_fallback=True,
        split_by_whitespace=False,
        character_coverage=character_coverage,
        max_sentence_length=20000,  # the default 4192 bytes would skip our longest lines
        num_threads=os.cpu_count(),
        minloglevel=2,
    )
    return output / "unigram.model"


# Checks the round trip on the tricky strings above and on real dev lines.
def check_lossless(model_path: Path, input_dir: str, lines_per_language: int = 300) -> None:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    failures = [text for text in LOSSLESS_TESTS if sp.decode(sp.encode(text)) != text]
    checked = len(LOSSLESS_TESTS)
    for path in iter_split_files(input_dir, "dev"):
        lines = path.read_text(encoding="utf-8").split("\n")[:lines_per_language]
        failures += [line for line in lines if sp.decode(sp.encode(line)) != line]
        checked += len(lines)
    print(f"lossless check: {checked} strings, {len(failures)} failures")
    for text in failures[:5]:
        print("  failed:", repr(text[:80]))

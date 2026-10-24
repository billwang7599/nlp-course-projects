# Token-level n-gram language model, shared by part3_ngram_lm.py and part3.py.
#
#   - every line is encoded with my Part-1 BPE and the end-of-sequence token EOS is appended
#   - counts[k] maps each k-gram (a tuple of token ids) to how often it appears in the
#     training lines; a k-gram never crosses a line boundary
#   - P(w | ctx) = (C(ctx + w) + alpha) / (C(ctx) + alpha * V), with no backoff
#     (alpha = 0 is plain maximum likelihood)
# This is the scoring rule written in part3_artifacts.py, so the TAs get the same numbers.

import math
from collections import Counter

from data import iter_split_files
from part3_artifacts import ngram_key

EOS = 50000  # one past the BPE vocabulary
VOCAB_SIZE = 50001  # 50,000 BPE tokens + EOS (no BOS is used)


# Returns {language: (encoded lines, number of characters)} for the cleaned files of one split.
# Each line is encoded on its own and ends with EOS, so no n-gram can cross a line.
def load_encoded(input_dir: str, split: str, tokenizer) -> dict[str, tuple[list[list[int]], int]]:
    result = {}
    for path in iter_split_files(input_dir, split):
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")[:-1]  # the file ends with a newline, so the last piece is empty
        result[path.parent.name] = ([tokenizer.encode(line) + [EOS] for line in lines], len(text))
    return result

def count_ngrams(lines: list[list[int]], max_order: int) -> dict[int, Counter]:
    orders = {}
    for i in range(1, max_order + 1):
        orders[i] = Counter()
    for line in lines:
        for i in range(len(line)):
            # k = length of the window starting at i; it must fit inside the line
            for k in range(1, min(max_order, len(line) - i) + 1):
                orders[k][tuple(line[i:i + k])] += 1

    return orders


def nll(counts: dict[int, Counter], lines: list[list[int]], n: int, alpha: float) -> tuple[float, tuple | None]:
    total_tokens = sum(counts[1].values())
    total_nll = 0.0
    for line in lines:
        for t, token in enumerate(line):
            m = min(t, n - 1)  # context length: shorter at the start of a line
            context = tuple(line[t - m:t])
            ngram = context + (token,)

            numerator = counts[m + 1].get(ngram, 0) + alpha
            if numerator == 0:
                return math.inf, ngram
            context_count = total_tokens if m == 0 else counts[m].get(context, 0)
            denominator = context_count + alpha * VOCAB_SIZE

            total_nll -= math.log(numerator / denominator)
    return total_nll, None


def perplexity(total_nll: float, n_tokens: int) -> float:
    return math.inf if total_nll == math.inf else math.exp(total_nll / n_tokens)


def bits_per_char(total_nll: float, n_chars: int) -> float:
    return total_nll / (n_chars * math.log(2))


# The dict that part3_artifacts.save_ngram_model expects: tuple keys become "12 123"
# strings, with a table for every order 1..n. alpha is saved in the file so that
# the TAs' scorer uses the same smoothing as I chose.
def to_model(counts: dict[int, Counter], n: int, alpha: float) -> dict:
    tables = {str(k): {ngram_key(gram): count for gram, count in counts[k].items()} for k in range(1, n + 1)}
    return {"n": n, "alpha": alpha, "vocab_size": VOCAB_SIZE, "counts": tables}

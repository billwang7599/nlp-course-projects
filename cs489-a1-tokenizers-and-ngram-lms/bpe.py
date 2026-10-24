import heapq
import re
import time
from array import array
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

DEAD = 0xFFFFFFFF  # id of a node that was merged into its left neighbor


class BPETokenizer:
    def __init__(self, vocab: list[bytes], merges: list[tuple[bytes, bytes]]) -> None:
        self.vocab = vocab
        self.merges = merges
        self.token_id = {token: i for i, token in enumerate(self.vocab)}
        self.rank = {(self.token_id[a], self.token_id[b]): r for r, (a, b) in enumerate(self.merges)}

    def encode(self, text: str) -> list[int]:
        byte_chain = TokenChain(text.encode("utf-8"))
        heap = [(self.rank[pair], pair) for pair, _ in byte_chain.counts.items() if pair in self.rank] # what to merge next
        heapq.heapify(heap)
        while heap:
            _, pair = heapq.heappop(heap)
            if pair not in byte_chain.counts:
                continue
            id1, id2 = pair
            id3 = self.token_id[self.vocab[id1] + self.vocab[id2]]
            new_pairs = byte_chain.merge(id1, id2, id3)
            for new_pair in new_pairs:
                if new_pair in self.rank:
                    heapq.heappush(heap, (self.rank[new_pair], new_pair))
        return byte_chain.token_ids()

    def decode(self, ids: list[int]) -> bytes:
        return b"".join(self.vocab[token_id] for token_id in ids)

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        directory = Path(directory)
        vocab = [bytes.fromhex(line) for line in (directory / "bpe_vocab.txt").read_text(encoding="ascii").splitlines()]
        merges = []
        for line in (directory / "bpe_merges.txt").read_text(encoding="ascii").splitlines():
            left, right = line.split()
            merges.append((bytes.fromhex(left), bytes.fromhex(right)))
        return cls(vocab, merges)

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bpe_vocab.txt").write_text("".join(t.hex() + "\n" for t in self.vocab), encoding="ascii")
        (out / "bpe_merges.txt").write_text(
            "".join(f"{a.hex()} {b.hex()}\n" for a, b in self.merges), encoding="ascii"
        )


class TokenChain:
    def __init__(self, text_bytes: bytes, can_merge: bytearray | None = None) -> None:
        n = len(text_bytes)
        if can_merge is None:
            can_merge = bytearray([1]) * max(n - 1, 0)
        self._ids = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.uint32)
        self._next_nodes = np.full(n, -1, dtype=np.int32)
        self._prev_nodes = np.full(n, -1, dtype=np.int32)
        self._id_at = memoryview(self._ids)
        self._next_of = memoryview(self._next_nodes)
        self._prev_of = memoryview(self._prev_nodes)
        for p in range(n - 1):
            if can_merge[p]:
                self._next_of[p] = p + 1
                self._prev_of[p + 1] = p
        self.counts: dict[tuple[int, int], int] = {}
        self._positions: dict[tuple[int, int], array] = {}
        for p in range(n):
            q = self._next_of[p]
            if q != -1:
                self._add_pair((self._id_at[p], self._id_at[q]), p)

    # Counts one more occurrence of the pair, which starts at node `position`.
    def _add_pair(self, pair: tuple[int, int], position: int) -> None:
        self.counts[pair] = self.counts.get(pair, 0) + 1
        self._positions.setdefault(pair, array("I")).append(position)

    # Replaces every (a, b) with new_id, left to right, updating the counts of
    # the pairs next to each merged spot. Returns the pairs whose count
    # changed and that still occur.
    def merge(self, left_id: int, right_id: int, new_id: int) -> set[tuple[int, int]]:
        changed = set()
        for p in sorted(self._positions.pop((left_id, right_id))):
            if self._id_at[p] != left_id:
                continue # stale
            q = self._next_of[p]
            if q == -1 or self._id_at[q] != right_id:
                continue  # stale
            x = self._prev_of[p]
            y = self._next_of[q]
            if x != -1:  # pair (x, left_id) becomes (x, new_id)
                cx = self._id_at[x]
                self.counts[(cx, left_id)] -= 1
                changed.add((cx, left_id))
                self._add_pair((cx, new_id), x)
                changed.add((cx, new_id))
            if y != -1:  # pair (right_id, y) becomes (new_id, y)
                cy = self._id_at[y]
                self.counts[(right_id, cy)] -= 1
                changed.add((right_id, cy))
                self._add_pair((new_id, cy), p)
                changed.add((new_id, cy))
                self._prev_of[y] = p
            self._id_at[p] = new_id
            self._id_at[q] = DEAD
            self._next_of[p] = y
        del self.counts[(left_id, right_id)]  # every occurrence of this pair is gone now
        changed.discard((left_id, right_id))
        vanished = {pair for pair in changed if self.counts[pair] == 0}
        for pair in vanished:
            del self.counts[pair]
            self._positions.pop(pair, None)
        return changed - vanished

    # The token ids of the live nodes, in text order.
    def token_ids(self) -> list[int]:
        return [token_id for token_id in self._id_at if token_id != DEAD]

    def token_count(self) -> int:
        return int(np.count_nonzero(self._ids != DEAD))


class BPETrainer:
    def __init__(self, vocab_size: int = 50000, barriers: Iterable[str] = ()) -> None:
        self.vocab_size = vocab_size
        self.barriers = frozenset(barriers)
        if any(not isinstance(b, str) or b == "" for b in self.barriers):
            raise ValueError("barriers must be non-empty strings")
        self.vocab = [bytes([i]) for i in range(256)]
        self.merges: list[tuple[bytes, bytes]] = []

    def train(self, text: str) -> BPETokenizer:
        text_bytes = text.encode("utf-8")
        self._chain = TokenChain(text_bytes, self._mergeable_gaps(text_bytes))
        # heap: (-count, pair), so the most frequent pair, then the smallest ids, pops first
        self._heap = [(-count, pair) for pair, count in self._chain.counts.items()]
        heapq.heapify(self._heap)
        t0 = time.time()
        while len(self.vocab) < self.vocab_size:
            pair = self._pop_best_pair()
            if pair is None:
                print(f"no pairs left, stopping at {len(self.vocab)} tokens")
                break
            new_id = len(self.vocab)
            self._add_token(pair)
            changed = self._chain.merge(pair[0], pair[1], new_id)
            self._refresh_heap(changed)
            if new_id % 1000 == 0:
                print(f"{new_id} tokens, {time.time() - t0:.0f}s")
        self._report_ratio(len(text_bytes))
        del self._chain, self._heap  # free the corpus
        return BPETokenizer(self.vocab, self.merges)

    # can_merge[p] is 1 if the bytes at p and p + 1 may end up in one token,
    # and 0 if a barrier boundary is between them.
    def _mergeable_gaps(self, text_bytes: bytes) -> bytearray:
        can_merge = bytearray([1]) * max(len(text_bytes) - 1, 0)
        for start, end in self._barrier_spans(text_bytes):
            if start > 0:
                can_merge[start - 1] = 0  # cut just before the barrier
            if end - 1 < len(can_merge):
                can_merge[end - 1] = 0  # cut just after it
        return can_merge

    # (start, end) of every occurrence of a barrier, left to right, non-overlapping
    # (end is exclusive). Longer barriers are tried first. UTF-8 is
    # self-synchronizing, so a whole character's bytes can only match where that
    # character really is.
    def _barrier_spans(self, text_bytes: bytes) -> Iterator[tuple[int, int]]:
        if not self.barriers:
            return
        alternatives = [re.escape(b.encode("utf-8")) for b in sorted(self.barriers, key=len, reverse=True)]
        for match in re.finditer(b"|".join(alternatives), text_bytes):
            yield match.span()

    # The best pair is the first heap entry whose count is still current;
    # older entries for a pair whose count changed are skipped (lazy deletion).
    def _pop_best_pair(self) -> tuple[int, int] | None:
        while self._heap:
            neg_count, pair = heapq.heappop(self._heap)
            if self._chain.counts.get(pair) == -neg_count:
                return pair
        return None

    def _add_token(self, pair: tuple[int, int]) -> None:
        a, b = pair
        self.vocab.append(self.vocab[a] + self.vocab[b])
        self.merges.append((self.vocab[a], self.vocab[b]))

    def _refresh_heap(self, changed: set[tuple[int, int]]) -> None:
        for pair in changed:
            heapq.heappush(self._heap, (-self._chain.counts[pair], pair))

    def _report_ratio(self, n_bytes: int) -> None:
        n_tokens = self._chain.token_count()
        print(f"corpus: {n_bytes} bytes -> {n_tokens} tokens (ratio {n_tokens / max(n_bytes, 1):.4f})")

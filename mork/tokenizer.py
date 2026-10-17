"""Byte-level BPE tokenizer with GPT-2-style pretokenization."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# Role markers first, then end/pad so those two ids stay last.
SPECIAL_TOKENS = ("<|user|>", "<|assistant|>", "<|end|>", "<|pad|>")

# OpenAI GPT-2 pretokenizer: merges never cross these piece boundaries.
GPT2_PRETOKENIZE_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


def _apply_merges_fast(ids: list[int], ranks: dict[tuple[int, int], int]) -> list[int]:
    """Greedily apply lowest-rank merge until no ranked pair remains."""
    if len(ids) < 2 or not ranks:
        return ids
    while True:
        best_rank = None
        best_pair = None
        for i in range(len(ids) - 1):
            rank = ranks.get((ids[i], ids[i + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_pair = (ids[i], ids[i + 1])
        if best_pair is None:
            return ids
        a, b = best_pair
        assert best_rank is not None
        new_id = 256 + best_rank
        out: list[int] = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out


def _apply_merges_ordered(ids: list[int], merges: list[tuple[int, int]]) -> list[int]:
    """Reference encoder: apply merges in creation order (slow)."""
    if len(ids) < 2:
        return ids
    current = ids
    other: list[int] = []
    for merge_idx, (a, b) in enumerate(merges):
        new_id = 256 + merge_idx
        n = len(current)
        found = False
        for j in range(n - 1):
            if current[j] == a and current[j + 1] == b:
                found = True
                break
        if not found:
            continue
        other.clear()
        j = 0
        while j < n:
            if j < n - 1 and current[j] == a and current[j + 1] == b:
                other.append(new_id)
                j += 2
            else:
                other.append(current[j])
                j += 1
        current, other = other, current
    return current


@dataclass
class ByteBPETokenizer:
    merges: list[tuple[int, int]]
    vocab_size: int
    pretokenize: bool = False
    train_corpus_chars: int | None = None
    special_tokens: tuple[str, ...] = field(default_factory=lambda: SPECIAL_TOKENS)

    def __post_init__(self) -> None:
        self.special_tokens = tuple(self.special_tokens)
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        # Longest first so multi-character specials win over prefixes.
        self._specials_by_length = sorted(self.special_tokens, key=len, reverse=True)

    def special_token_id(self, name: str) -> int:
        return 256 + len(self.merges) + self.special_tokens.index(name)

    def _encode_ordinary(self, text: str, *, reference: bool) -> list[int]:
        if not text:
            return []
        if self.pretokenize:
            out: list[int] = []
            for piece in GPT2_PRETOKENIZE_PATTERN.findall(text):
                ids = list(piece.encode("utf-8"))
                if reference:
                    out.extend(_apply_merges_ordered(ids, self.merges))
                else:
                    out.extend(_apply_merges_fast(ids, self._ranks))
            return out
        ids = list(text.encode("utf-8"))
        if reference:
            return _apply_merges_ordered(ids, self.merges)
        return _apply_merges_fast(ids, self._ranks)

    def encode(self, text: str, *, reference: bool = False) -> list[int]:
        if not text:
            return []

        out: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched: str | None = None
            for special in self._specials_by_length:
                if text.startswith(special, i):
                    matched = special
                    break
            if matched is not None:
                out.append(self.special_token_id(matched))
                i += len(matched)
                continue

            next_special = n
            for special in self._specials_by_length:
                j = text.find(special, i)
                if j != -1 and j < next_special:
                    next_special = j
            out.extend(self._encode_ordinary(text[i:next_special], reference=reference))
            i = next_special
        return out

    def decode(self, ids: list[int]) -> str:
        bytes_out = bytearray()
        for token_id in ids:
            bytes_out.extend(self._token_bytes(token_id))
        return bytes_out.decode("utf-8", errors="replace")

    def _token_bytes(self, token_id: int) -> bytes:
        if token_id < 256:
            return bytes([token_id])
        merge_idx = token_id - 256
        if merge_idx < len(self.merges):
            return self._token_bytes(self.merges[merge_idx][0]) + self._token_bytes(
                self.merges[merge_idx][1]
            )
        special_start = 256 + len(self.merges)
        special_idx = token_id - special_start
        if 0 <= special_idx < len(self.special_tokens):
            return self.special_tokens[special_idx].encode("utf-8")
        raise KeyError(f"unknown token id: {token_id}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": list(self.special_tokens),
            "pretokenize": self.pretokenize,
            "train_corpus_chars": self.train_corpus_chars,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in payload["merges"]]
        specials = payload.get("special_tokens")
        if specials is None:
            specials = list(SPECIAL_TOKENS)
        return cls(
            merges=merges,
            vocab_size=payload["vocab_size"],
            pretokenize=bool(payload.get("pretokenize", False)),
            train_corpus_chars=payload.get("train_corpus_chars"),
            special_tokens=tuple(specials),
        )


def train_pretokenized_bpe(
    text: str,
    vocab_size: int,
    *,
    pattern: re.Pattern[str] = GPT2_PRETOKENIZE_PATTERN,
) -> ByteBPETokenizer:
    """Train BPE on GPT-2-style pretokens using the full provided corpus.

    Merges only occur inside a pretoken (never across spaces/word boundaries).
    ``<|user|>``, ``<|assistant|>``, ``<|end|>``, and ``<|pad|>`` are reserved
    ids after the merges.
    """
    specials = SPECIAL_TOKENS
    min_vocab = 256 + len(specials)
    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size must be at least {min_vocab}")

    word_counts: Counter[bytes] = Counter()
    for piece in pattern.findall(text):
        if piece:
            word_counts[piece.encode("utf-8")] += 1

    num_merges = vocab_size - min_vocab
    splits: dict[bytes, list[int]] = {word: list(word) for word in word_counts}
    counts = dict(word_counts)

    pair_stats: Counter[tuple[int, int]] = Counter()
    pair_words: dict[tuple[int, int], set[bytes]] = defaultdict(set)
    for word, sp in splits.items():
        c = counts[word]
        for i in range(len(sp) - 1):
            pair = (sp[i], sp[i + 1])
            pair_stats[pair] += c
            pair_words[pair].add(word)

    merges: list[tuple[int, int]] = []
    log_every = max(1, num_merges // 20)

    for merge_idx in range(num_merges):
        if not pair_stats:
            break
        best = max(pair_stats, key=lambda p: (pair_stats[p], p))
        best_count = pair_stats[best]
        if best_count <= 0:
            break

        a, b = best
        merges.append(best)
        affected = list(pair_words.get(best, ()))

        for word in affected:
            sp = splits[word]
            c = counts[word]
            for i in range(len(sp) - 1):
                pair_stats[(sp[i], sp[i + 1])] -= c

            out: list[int] = []
            i = 0
            n = len(sp)
            while i < n:
                if i < n - 1 and sp[i] == a and sp[i + 1] == b:
                    out.append(256 + len(merges) - 1)
                    i += 2
                else:
                    out.append(sp[i])
                    i += 1
            splits[word] = out

            for i in range(len(out) - 1):
                pair = (out[i], out[i + 1])
                pair_stats[pair] += c
                pair_words[pair].add(word)

        for pair, score in list(pair_stats.items()):
            if score <= 0:
                del pair_stats[pair]
                pair_words.pop(pair, None)

        if merge_idx % log_every == 0 or merge_idx == num_merges - 1:
            print(
                f"  merge {merge_idx + 1:,}/{num_merges:,} "
                f"({100 * (merge_idx + 1) / num_merges:.0f}%) "
                f"count={best_count:,}",
                flush=True,
            )

    print(
        f"  pretok types={len(word_counts):,}  "
        f"corpus_chars={len(text):,}  merges={len(merges):,}  "
        f"specials={len(specials)}",
        flush=True,
    )
    return ByteBPETokenizer(
        merges=merges,
        vocab_size=vocab_size,
        pretokenize=True,
        train_corpus_chars=len(text),
        special_tokens=specials,
    )

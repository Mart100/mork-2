"""Byte-level BPE backed by Hugging Face tokenizers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# Reserved from the start so pretrain and SFT share one vocab.
SPECIAL_TOKENS = ("<|user|>", "<|assistant|>", "<|end|>", "<|pad|>")
DEFAULT_VOCAB_SIZE = 16_384


def _special_tokens() -> list[AddedToken]:
    return [
        AddedToken(name, special=True, normalized=False, lstrip=False, rstrip=False)
        for name in SPECIAL_TOKENS
    ]


def _new_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def _trainer(vocab_size: int) -> BpeTrainer:
    min_vocab = 256 + len(SPECIAL_TOKENS)
    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size must be at least {min_vocab}")
    return BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=_special_tokens(),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
        min_frequency=2,
    )


class MorkTokenizer:
    """Encode and decode text. Specials in the string stay single tokens."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, path: str | Path) -> MorkTokenizer:
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def special_token_id(self, name: str) -> int:
        token_id = self.tokenizer.token_to_id(name)
        if token_id is None:
            raise KeyError(name)
        return token_id

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


def train_tokenizer(
    files: Iterable[str | Path],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> MorkTokenizer:
    """Train on newline-delimited text files."""
    tokenizer = _new_tokenizer()
    tokenizer.train([str(path) for path in files], _trainer(vocab_size))
    return MorkTokenizer(tokenizer)


def train_tokenizer_from_iterator(
    texts: Iterable[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    *,
    length: int | None = None,
) -> MorkTokenizer:
    """Train on an iterable of documents (one string per document)."""
    tokenizer = _new_tokenizer()
    tokenizer.train_from_iterator(texts, trainer=_trainer(vocab_size), length=length)
    return MorkTokenizer(tokenizer)

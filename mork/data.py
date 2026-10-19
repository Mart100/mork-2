import os
from pathlib import Path

import numpy as np
import torch


def load_tokens(filename: str | os.PathLike[str]) -> torch.Tensor:
    npt = np.load(filename)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    def __init__(
        self,
        B: int,
        T: int,
        split: str,
        data_dir: Path,
        *,
        shuffle_chunk_tokens: int = 0,
        fixed_shuffle: bool = False,
        shuffle_seed: int = 0,
    ):
        self.B = B
        self.T = T
        assert split in {"train", "val"}

        data_root = Path(data_dir)
        # Matches e.g. corpus_train_000.npy / corpus_val_000.npy
        shards = sorted(data_root.glob(f"*_{split}_*.npy"))
        self.shards = [str(s) for s in shards]
        assert len(self.shards) > 0, (
            f"no shards found for split {split} in {data_root} "
            f"(expected *_{split}_*.npy)"
        )

        # Shuffle large contiguous chunks so local token order stays intact.
        # Training draws a new order each pass. Validation keeps one seeded
        # order, so reset() always returns the same mixed slice.
        self.fixed_shuffle = fixed_shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_chunk_tokens = shuffle_chunk_tokens if shuffle_chunk_tokens > 0 else 0
        self.reshuffle_on_wrap = self.shuffle_chunk_tokens > 0 and not fixed_shuffle
        self._base_tokens: torch.Tensor | None = None
        self._chunk_perm: list[int] | None = None
        self._perms: dict[int, list[int] | None] = {}

        self.current_shard = 0
        self._load_shard(self.current_shard, reshuffle=True)

        print(
            f"loaded {len(self._base_tokens):,} tokens for split {split} "
            f"({len(self.shards)} shard(s)"
            + (
                f", {'fixed ' if self.fixed_shuffle else ''}shuffle chunks of "
                f"{self.shuffle_chunk_tokens:,} tokens)"
                if self.shuffle_chunk_tokens
                else ")"
            )
        )

    def _load_shard(self, shard: int, *, reshuffle: bool) -> None:
        self.current_shard = shard
        self._base_tokens = load_tokens(self.shards[shard])
        if reshuffle:
            self._reshuffle()
        else:
            self.tokens = self._apply_perm(self._chunk_perm)
        self.current_position = 0

    def _reshuffle(self) -> None:
        assert self._base_tokens is not None
        n = self.shuffle_chunk_tokens
        if n <= 0:
            self._chunk_perm = None
            self.tokens = self._base_tokens
            return

        num_full = len(self._base_tokens) // n
        if num_full <= 1:
            self._chunk_perm = None
            self._perms[self.current_shard] = None
            self.tokens = self._base_tokens
            return

        if self.fixed_shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.shuffle_seed + self.current_shard)
            perm = torch.randperm(num_full, generator=generator)
        else:
            perm = torch.randperm(num_full)
        self._chunk_perm = perm.tolist()
        self._perms[self.current_shard] = self._chunk_perm
        self.tokens = self._apply_perm(self._chunk_perm)
        print(
            f"shuffled {num_full} chunks "
            f"({n:,} tokens each, +{len(self._base_tokens) % n:,} remainder)"
        )

    def _apply_perm(self, perm: list[int] | None) -> torch.Tensor:
        assert self._base_tokens is not None
        n = self.shuffle_chunk_tokens
        if perm is None or n <= 0:
            return self._base_tokens

        chunks = [self._base_tokens[i * n : (i + 1) * n] for i in perm]
        remainder = self._base_tokens[len(perm) * n :]
        if len(remainder) > 0:
            chunks.append(remainder)
        return torch.cat(chunks)

    def reset(self) -> None:
        """Rewind to the first batch. With ``fixed_shuffle`` the order is identical each time."""
        if self.current_shard != 0:
            self._chunk_perm = self._perms.get(0)
            self._load_shard(0, reshuffle=0 not in self._perms)
        self.current_position = 0

    def state_dict(self) -> dict:
        return {
            "current_shard": self.current_shard,
            "current_position": self.current_position,
            "shuffle_chunk_tokens": self.shuffle_chunk_tokens,
            "chunk_perm": self._chunk_perm,
        }

    def load_state_dict(self, state: dict) -> None:
        shard = int(state["current_shard"])
        position = int(state["current_position"])
        if not (0 <= shard < len(self.shards)):
            raise ValueError(
                f"checkpoint shard index {shard} out of range "
                f"(have {len(self.shards)} shards)"
            )

        saved_chunk = int(state.get("shuffle_chunk_tokens", 0))
        if self.fixed_shuffle and saved_chunk == 0:
            print(
                "checkpoint validation cursor has no shuffle; "
                "scoring the shuffled holdout from the start"
            )
            return
        if saved_chunk != self.shuffle_chunk_tokens:
            print(
                f"warning: checkpoint shuffle_chunk_tokens={saved_chunk} "
                f"!= current {self.shuffle_chunk_tokens}; "
                f"restoring with checkpoint value for this resume"
            )
            self.shuffle_chunk_tokens = saved_chunk

        perm = state.get("chunk_perm")
        self._chunk_perm = list(perm) if perm is not None else None
        self._perms[shard] = self._chunk_perm
        self._load_shard(shard, reshuffle=False)

        if not (0 <= position < len(self.tokens)):
            raise ValueError(
                f"checkpoint position {position} out of range "
                f"for shard length {len(self.tokens)}"
            )
        self.current_position = position
        print(
            f"restored {self.shards[self.current_shard]!r} loader "
            f"at shard={self.current_shard} position={self.current_position:,}"
            + (
                f" (shuffled {len(self._chunk_perm)} chunks)"
                if self._chunk_perm is not None
                else ""
            )
        )

    def _advance_shard(self) -> None:
        nxt = (self.current_shard + 1) % len(self.shards)
        if self.reshuffle_on_wrap:
            self._load_shard(nxt, reshuffle=True)
            return
        # Fixed order: reuse the permutation from the first visit.
        known = nxt in self._perms
        self._chunk_perm = self._perms.get(nxt)
        self._load_shard(nxt, reshuffle=not known)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        needed = B * T + 1
        if self.current_position + needed > len(self.tokens):
            self._advance_shard()
            if needed > len(self.tokens):
                raise RuntimeError(
                    f"shard {self.shards[self.current_shard]!r} has "
                    f"{len(self.tokens)} tokens, need at least {needed}"
                )

        buf = self.tokens[self.current_position : self.current_position + needed]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        return x, y

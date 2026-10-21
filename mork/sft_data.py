"""Pack SFT dialogues into the trainer's fixed windows.

Each row becomes ``<|user|> ... <|assistant|> ... <|end|>``. Loss is on the
assistant reply and on ``<|end|>``. Windows are 512 tokens. A dialogue stays
in one window; leftover slots are ``<|pad|>`` and ignored by the loss. The
token after a dialogue does not predict the next dialogue.

Train reshuffles examples every epoch. Validation is packed once.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mork.config import PROJECT_ROOT, TrainConfig
from mork.sft_prepare import format_dialog
from mork.tokenizer import MorkTokenizer

IGNORE_INDEX = -100
ASSISTANT_MARK = "<|assistant|>\n"


def loss_targets(
    ids: list[int],
    offsets: list[tuple[int, int]],
    boundary: int,
    end_id: int,
) -> list[int]:
    """Target at position i is the next token when that token is part of the reply.

    The last prompt token predicts the first reply token. ``<|end|>`` is a
    target. The position of ``<|end|>`` itself is ignored, so a following
    dialogue is not predicted. A token that straddles the reply boundary
    counts as reply text.
    """
    targets = [IGNORE_INDEX] * len(ids)
    for i in range(len(ids) - 1):
        nxt = i + 1
        start, end = offsets[nxt]
        straddles = start < boundary < end
        # (0, 0) is how special tokens report "no span". <|end|> is matched by id.
        in_reply = start >= boundary and (start, end) != (0, 0)
        if ids[nxt] == end_id or straddles or in_reply:
            targets[i] = ids[nxt]
    return targets


def encode_example(
    tokenizer: MorkTokenizer,
    user: str,
    assistant: str,
) -> tuple[list[int], list[int]]:
    full = format_dialog(user, assistant)
    boundary = full.rindex(ASSISTANT_MARK) + len(ASSISTANT_MARK)
    enc = tokenizer.tokenizer.encode(full)
    ids = list(enc.ids)
    targets = loss_targets(
        ids,
        list(enc.offsets),
        boundary,
        tokenizer.special_token_id("<|end|>"),
    )
    return ids, targets


def encode_rows(
    tokenizer: MorkTokenizer,
    rows: list[dict],
    *,
    seq_len: int,
    batch_size: int = 2048,
) -> list[tuple[list[int], list[int]]]:
    end_id = tokenizer.special_token_id("<|end|>")
    examples: list[tuple[list[int], list[int]]] = []
    too_long = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        texts: list[str] = []
        boundaries: list[int] = []
        for row in chunk:
            full = format_dialog(row["user"], row["assistant"])
            texts.append(full)
            boundaries.append(full.rindex(ASSISTANT_MARK) + len(ASSISTANT_MARK))
        for enc, boundary in zip(tokenizer.tokenizer.encode_batch(texts), boundaries, strict=True):
            ids = list(enc.ids)
            if len(ids) > seq_len:
                too_long += 1
                continue
            examples.append((ids, loss_targets(ids, list(enc.offsets), boundary, end_id)))
        done = min(start + batch_size, len(rows))
        if done == len(rows) or done % 20_000 == 0:
            print(f"  encoded {done:,} / {len(rows):,} rows", flush=True)
    if too_long:
        raise SystemExit(f"{too_long} SFT rows are longer than {seq_len} tokens")
    if not examples:
        raise SystemExit("SFT split produced no examples")
    return examples


def load_rows(sft_dir: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    paths = sorted(sft_dir.glob(f"*_{split}.jsonl"))
    if not paths:
        raise SystemExit(f"no *_{split}.jsonl files in {sft_dir}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no SFT rows for split {split} in {sft_dir}")
    return rows


def pack_windows(
    examples: list[tuple[list[int], list[int]]],
    seq_len: int,
    pad_id: int,
) -> tuple[list[int], list[int]]:
    """Place whole dialogues into ``seq_len`` windows. Does not split one."""
    tokens: list[int] = []
    targets: list[int] = []
    window_tokens = [pad_id] * seq_len
    window_targets = [IGNORE_INDEX] * seq_len
    filled = 0

    def flush() -> None:
        nonlocal filled, window_tokens, window_targets
        tokens.extend(window_tokens)
        targets.extend(window_targets)
        window_tokens = [pad_id] * seq_len
        window_targets = [IGNORE_INDEX] * seq_len
        filled = 0

    for ids, example_targets in examples:
        if len(ids) > seq_len:
            raise ValueError(f"dialogue has {len(ids)} tokens, window is {seq_len}")
        if len(ids) != len(example_targets):
            raise ValueError("token and target lengths differ")
        if filled and filled + len(ids) > seq_len:
            flush()
        end = filled + len(ids)
        window_tokens[filled:end] = ids
        window_targets[filled:end] = example_targets
        filled = end
    if filled:
        flush()
    return tokens, targets


def truncate_block(
    tokens: list[int],
    targets: list[int],
    block_tokens: int,
) -> tuple[list[int], list[int]]:
    """Drop a trailing partial optimizer step so every micro-batch divides the stream."""
    usable = (len(tokens) // block_tokens) * block_tokens
    if usable <= 0:
        raise SystemExit(
            f"SFT stream has {len(tokens):,} tokens, need at least {block_tokens:,}"
        )
    return tokens[:usable], targets[:usable]


class SftLoader:
    """Yields ``(x, y)`` with ``y`` aligned to ``x``. Ignored positions are -100.

    ``.B`` is writable so the auto-batch probe can shrink the micro-batch.
    """

    def __init__(
        self,
        examples: list[tuple[list[int], list[int]]],
        *,
        B: int,
        T: int,
        split: str,
        seed: int,
        pad_id: int,
        block_tokens: int,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be train or val, got {split!r}")
        self.examples = examples
        self.B = B
        self.T = T
        self.split = split
        self.seed = seed
        self.pad_id = pad_id
        self.block_tokens = block_tokens
        self.epoch = 0
        self.pos = 0
        self.tokens = torch.empty(0, dtype=torch.long)
        self.targets = torch.empty(0, dtype=torch.long)
        self._pack()

    def _order(self) -> list[int]:
        generator = torch.Generator()
        # Validation stays on one order. Train uses a new order each epoch.
        generator.manual_seed(self.seed if self.split == "val" else self.seed + self.epoch)
        return torch.randperm(len(self.examples), generator=generator).tolist()

    def _pack(self) -> None:
        ordered = [self.examples[i] for i in self._order()]
        tokens, targets = pack_windows(ordered, self.T, self.pad_id)
        tokens, targets = truncate_block(tokens, targets, self.block_tokens)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def reset(self) -> None:
        # Epoch 0 is already the packed order. A probe only moves ``pos``.
        repack = self.epoch != 0
        self.epoch = 0
        self.pos = 0
        if repack:
            self._pack()

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "split": self.split}

    def load_state_dict(self, state: dict) -> None:
        if state.get("split") not in (None, self.split):
            raise ValueError(
                f"checkpoint loader split {state.get('split')!r} != {self.split!r}"
            )
        self.epoch = int(state["epoch"])
        self.pos = int(state["pos"])
        self._pack()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        need = self.B * self.T
        if self.pos + need > len(self.tokens):
            if self.split == "val":
                raise RuntimeError(
                    f"SFT val loader exhausted at token {self.pos:,} "
                    f"of {len(self.tokens):,} (need {need:,})"
                )
            self.epoch += 1
            self.pos = 0
            self._pack()
        x = self.tokens[self.pos : self.pos + need].view(self.B, self.T)
        y = self.targets[self.pos : self.pos + need].view(self.B, self.T)
        self.pos += need
        return x, y


def build_sft_loaders(
    tokenizer: MorkTokenizer,
    train_cfg: TrainConfig,
) -> tuple[SftLoader, SftLoader]:
    sft_dir = train_cfg.data_dir / "sft"
    pad_id = tokenizer.special_token_id("<|pad|>")
    loaders: list[SftLoader] = []
    for split in ("train", "val"):
        print(f"loading SFT {split} from {sft_dir}", flush=True)
        rows = load_rows(sft_dir, split)
        examples = encode_rows(tokenizer, rows, seq_len=train_cfg.seq_len)
        loader = SftLoader(
            examples,
            B=train_cfg.batch_size,
            T=train_cfg.seq_len,
            split=split,
            seed=train_cfg.seed,
            pad_id=pad_id,
            block_tokens=train_cfg.total_batch_size,
        )
        print(
            f"  {split}: {len(examples):,} dialogues, "
            f"{len(loader.tokens):,} packed tokens"
        )
        loaders.append(loader)
    train_loader, val_loader = loaders
    return train_loader, val_loader


def _self_check() -> None:
    import tempfile

    from mork.evaluate import extract_source, instruction_from_stub, run_program, smoke_tasks

    # A dialogue that does not fit in the remainder starts at the next window,
    # and the boundary target does not predict the next dialogue.
    first = ([1, 2, 3], [IGNORE_INDEX, 3, IGNORE_INDEX])
    second = ([4, 5], [5, IGNORE_INDEX])
    tokens, targets = pack_windows([first, second], seq_len=8, pad_id=0)
    assert tokens[:5] == [1, 2, 3, 4, 5]
    assert targets[:5] == [IGNORE_INDEX, 3, IGNORE_INDEX, 5, IGNORE_INDEX]
    assert tokens[5:] == [0, 0, 0]
    assert targets[5:] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]

    long = ([7] * 400, [IGNORE_INDEX] * 400)
    short = ([8] * 200, [IGNORE_INDEX] * 200)
    tokens, targets = pack_windows([long, short], seq_len=512, pad_id=0)
    assert tokens[400:512] == [0] * 112
    assert targets[400:512] == [IGNORE_INDEX] * 112
    assert tokens[512:712] == [8] * 200
    assert len(tokens) == 1024

    tiny = [([1, 2, 3], [IGNORE_INDEX, 3, IGNORE_INDEX])] * 10
    loader = SftLoader(
        tiny, B=1, T=8, split="train", seed=0, pad_id=0, block_tokens=16
    )
    x1, y1 = loader.next_batch()
    state = loader.state_dict()
    x2, y2 = loader.next_batch()
    loader.load_state_dict(state)
    x3, y3 = loader.next_batch()
    assert torch.equal(x2, x3) and torch.equal(y2, y3)
    loader.reset()
    x0, _ = loader.next_batch()
    assert torch.equal(x1, x0)
    assert int(x1[0, -1]) == 0 and int(y1[0, -1]) == IGNORE_INDEX
    val = SftLoader(tiny, B=1, T=8, split="val", seed=1, pad_id=0, block_tokens=16)
    frozen = val.tokens.clone()
    val.reset()
    assert torch.equal(frozen, val.tokens)

    tokenizer_path = PROJECT_ROOT / "data" / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise SystemExit(f"tokenizer not found: {tokenizer_path}")
    tokenizer = MorkTokenizer.load(tokenizer_path)
    user = "Write a function that adds two numbers."
    assistant = "def add(a, b):\n    return a + b"
    ids, row_targets = encode_example(tokenizer, user, assistant)
    assert len(ids) == len(row_targets)
    assert row_targets[-1] == IGNORE_INDEX
    supervised = [token for token in row_targets if token != IGNORE_INDEX]
    decoded = tokenizer.decode(supervised)
    assert decoded.endswith("<|end|>"), decoded
    body = decoded[: -len("<|end|>")].removeprefix("<|assistant|>").lstrip("\n")
    assert body == assistant, decoded
    assert user not in decoded, decoded
    first_supervised = next(i for i, token in enumerate(row_targets) if token != IGNORE_INDEX)
    assert all(token == IGNORE_INDEX for token in row_targets[:first_supervised])
    assert len(ids) <= 512

    task = next(item for item in smoke_tasks() if item.name == "add")
    instruction = instruction_from_stub(task.prompt)
    assert instruction == (
        "Write a Python function `add(a, b)`. Return the sum of a and b. "
        "Example: `add(2, 3)` returns `5`."
    )
    completions = [
        "def add(a, b):\n    return a + b\n<|end|>",
        "```python\ndef add(a, b):\n    return a + b\n```\n<|end|>",
        "Sure.\n\ndef add(a, b):\n    return a + b\nprint('no')\n<|end|>",
    ]
    with tempfile.TemporaryDirectory() as workdir:
        for completion in completions:
            source = extract_source(completion)
            assert "print" not in source
            compiled, passed = run_program(task.setup + source + "\n" + task.tests, workdir)
            assert compiled and passed, source
    print("sft_data self-check ok")


if __name__ == "__main__":
    _self_check()

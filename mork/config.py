from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    """~98M-param GPT (12 layers, width 768, vocab 16384) for 4 GB VRAM."""

    block_size: int = 512
    vocab_size: int = 16_384  # byte-level BPE, including the four special tokens
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    rope_theta: float = 10_000.0  # RoPE base frequency (LLaMA default)


@dataclass
class TrainConfig:
    """Training defaults for 4 GB VRAM: micro-batch 16, seq 512, 32k tokens/step.

    The auto-batch probe may shrink the micro-batch. ``max_steps`` is a short
    default; set the real token budget on the CLI once shards exist.
    """

    data_dir: Path = PROJECT_ROOT / "data"
    device: str = "cuda"
    seed: int = 1337

    batch_size: int = 16
    seq_len: int = 512
    total_batch_size: int = 32_768  # 2^15 tokens per optimizer step
    # Train-only: shuffle contiguous chunks each epoch (0 = disable).
    # Large chunks keep local context; ~32k ≈ one optimizer step of tokens.
    shuffle_chunk_tokens: int = 32_768

    max_lr: float = 8e-4  # peak LR from the 88M pretrain
    min_lr_ratio: float = 0.2
    warmup_steps: int = 100
    max_steps: int = 4000
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    compile: bool = True
    dtype: str = "bfloat16"

    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    checkpoint_every: int = 100
    snapshot_every: int = 1000
    resume: bool = True
    # Load model weights from this checkpoint, then start a new schedule at
    # step 0 with a fresh optimizer (ignored if resume finds latest).
    init_from: Path | None = None

    runs_dir: Path = PROJECT_ROOT / "runs"
    eval_every: int = 100
    eval_batches: int = 500

    # Start-of-run: shrink micro-batch (keep total_batch_size) if step time is too high.
    auto_batch_probe: bool = True
    auto_batch_max_dt_s: float = 3.0
    save_checkpoints: bool = True

    @property
    def grad_accum_steps(self) -> int:
        micro_batch_tokens = self.batch_size * self.seq_len
        assert self.total_batch_size % micro_batch_tokens == 0, (
            f"total_batch_size {self.total_batch_size} must be divisible by "
            f"batch_size * seq_len ({micro_batch_tokens})"
        )
        return self.total_batch_size // micro_batch_tokens

    @property
    def min_lr(self) -> float:
        return self.max_lr * self.min_lr_ratio

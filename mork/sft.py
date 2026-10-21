"""Supervised fine-tune on data/sft, full weights, same loop as pretraining.

The first run loads weights from --init-from and writes checkpoints/sft.
A later run resumes checkpoints/sft/latest.pt. The pretrain checkpoint
directory is left alone.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from mork.config import PROJECT_ROOT, ModelConfig, TrainConfig
from mork.sft_data import build_sft_loaders
from mork.tokenizer import MorkTokenizer
from mork.train import main as train_main

WARMUP_STEPS = 50
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tune Mork-2.")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help=(
            "Cooled checkpoint to load on the first run. Ignored once "
            "checkpoints/sft/latest.pt exists."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Passes over the packed training tokens (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--max-lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Peak learning rate (default: {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "sft",
        help="SFT checkpoint directory (default: checkpoints/sft).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory that contains sft/ and tokenizer.json (default: data).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (default: cuda).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoints/sft/latest.pt and start from --init-from.",
    )
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    latest = args.checkpoint_dir / "latest.pt"
    resuming = not args.fresh and latest.is_file()
    if not resuming and (args.init_from is None or not args.init_from.is_file()):
        raise SystemExit(
            "the first SFT run needs --init-from pointing at the cooled checkpoint. "
            f"Later runs resume {latest}."
        )

    train_cfg = replace(
        TrainConfig(),
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        device=args.device,
        max_lr=args.max_lr,
        resume=not args.fresh,
        compile=not args.no_compile,
        init_from=args.init_from,
        snapshot_every=400,
        warmup_steps=WARMUP_STEPS,
        min_lr_ratio=0.0,
    )
    tokenizer = MorkTokenizer.load(train_cfg.data_dir / "tokenizer.json")
    train_loader, val_loader = build_sft_loaders(tokenizer, train_cfg)
    steps_per_epoch = len(train_loader.tokens) // train_cfg.total_batch_size
    max_steps = args.epochs * steps_per_epoch
    if max_steps <= WARMUP_STEPS:
        raise SystemExit(
            f"schedule is {max_steps} steps, which does not leave room "
            f"for {WARMUP_STEPS} warmup steps"
        )
    cooldown_steps = max(1, max_steps // 4)
    cooldown_start = max_steps - cooldown_steps
    train_cfg = replace(
        train_cfg,
        max_steps=max_steps,
        cooldown_start=cooldown_start,
        cooldown_steps=cooldown_steps,
        eval_tokens=len(val_loader.tokens),
    )
    print(
        f"sft: {args.epochs} epochs, {steps_per_epoch:,} steps/epoch, "
        f"{max_steps:,} steps, lr {train_cfg.max_lr:.4e}, "
        f"warmup {WARMUP_STEPS}, cooldown at step {cooldown_start:,}, "
        f"val tokens {train_cfg.eval_tokens:,}"
    )
    train_main(
        train_cfg=train_cfg,
        model_cfg=ModelConfig(),
        train_loader=train_loader,
        val_loader=val_loader,
    )


if __name__ == "__main__":
    main()

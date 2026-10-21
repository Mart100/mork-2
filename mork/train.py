import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from mork.checkpoint import (
    best_val_checkpoint_path,
    latest_checkpoint_path,
    load_best_val_loss,
    load_checkpoint,
    load_weights_only,
    maybe_save_best_val,
    maybe_save_checkpoints,
    save_checkpoint,
    val_slice_tag,
)
from mork.config import ModelConfig, TrainConfig
from mork.data import DataLoaderLite
from mork.metrics import MetricsLogger, resolve_run_dir
from mork.model import GPT


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _set_micro_batch(
    train_cfg: TrainConfig,
    train_loader: DataLoaderLite,
    val_loader: DataLoaderLite,
    batch_size: int,
) -> None:
    """Set micro-batch while keeping total_batch_size (raises if not divisible)."""
    train_cfg.batch_size = batch_size
    train_loader.B = batch_size
    val_loader.B = batch_size
    # Validate via property (asserts divisibility).
    _ = train_cfg.grad_accum_steps


def _next_smaller_batch(train_cfg: TrainConfig, batch_size: int) -> int | None:
    """Largest batch_size < current that still divides total_batch_size."""
    for candidate in range(batch_size // 2, 0, -1):
        micro = candidate * train_cfg.seq_len
        if train_cfg.total_batch_size % micro == 0:
            return candidate
    return None


def _probe_step_dt_s(
    *,
    model: torch.nn.Module,
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoaderLite,
    train_cfg: TrainConfig,
    device: str,
    dtype: torch.dtype,
) -> float:
    """One full train step including optimizer.step (matches real VRAM use)."""
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(train_cfg.grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(x, y)
        assert loss is not None
        (loss / train_cfg.grad_accum_steps).backward()
    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_cfg.grad_clip)
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


def probe_auto_batch(
    *,
    model: torch.nn.Module,
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoaderLite,
    val_loader: DataLoaderLite,
    train_cfg: TrainConfig,
    device: str,
    dtype: torch.dtype,
    max_dt_s: float,
    measure_steps: int = 2,
) -> torch.optim.Optimizer:
    """Shrink micro-batch until full-step time is healthy (keeps total_batch_size).

    Must include ``optimizer.step()``: Adam moment buffers are allocated on the
    first step and are a large share of VRAM. Probe restores weights and returns
    a fresh optimizer so training still starts from the original init.
    """
    print(
        f"auto-batch probe: max_dt={max_dt_s:.1f}s, "
        f"starting batch_size={train_cfg.batch_size}, "
        f"grad_accum={train_cfg.grad_accum_steps}"
    )
    saved_weights = {
        name: tensor.detach().cpu().clone()
        for name, tensor in raw_model.state_dict().items()
    }

    try:
        while True:
            dts: list[float] = []
            oom = False
            try:
                # Warmup: compile + allocate Adam state (not timed for the decision).
                _probe_step_dt_s(
                    model=model,
                    raw_model=raw_model,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    train_cfg=train_cfg,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(measure_steps):
                    dts.append(
                        _probe_step_dt_s(
                            model=model,
                            raw_model=raw_model,
                            optimizer=optimizer,
                            train_loader=train_loader,
                            train_cfg=train_cfg,
                            device=device,
                            dtype=dtype,
                        )
                    )
            except torch.cuda.OutOfMemoryError:
                oom = True
                if device == "cuda":
                    torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)

            if not oom:
                median_dt = sorted(dts)[len(dts) // 2]
                print(
                    f"  probe batch_size={train_cfg.batch_size} "
                    f"accum={train_cfg.grad_accum_steps}: "
                    f"median_dt={median_dt:.2f}s "
                    f"(samples={[round(d, 2) for d in dts]})"
                )
                if median_dt <= max_dt_s:
                    print(
                        f"auto-batch probe ok: batch_size={train_cfg.batch_size}, "
                        f"grad_accum={train_cfg.grad_accum_steps}"
                    )
                    break

            old_bs = train_cfg.batch_size
            new_bs = _next_smaller_batch(train_cfg, old_bs)
            if new_bs is None:
                print(
                    f"auto-batch probe: giving up at batch_size={old_bs} "
                    f"({'OOM' if oom else f'dt>{max_dt_s}s'}); continuing anyway"
                )
                break

            _set_micro_batch(train_cfg, train_loader, val_loader, new_bs)
            if device == "cuda":
                torch.cuda.empty_cache()
            reason = "OOM" if oom else f"dt>{max_dt_s}s"
            print(
                f"  auto-batch: {old_bs} → {new_bs} "
                f"(accum {train_cfg.grad_accum_steps}) due to {reason}"
            )
    finally:
        raw_model.load_state_dict(saved_weights)
        optimizer = configure_optimizers(
            raw_model,
            weight_decay=train_cfg.weight_decay,
            learning_rate=train_cfg.max_lr,
        )
        if device == "cuda":
            torch.cuda.empty_cache()
        print("auto-batch probe: restored model weights, reset optimizer")

    return optimizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train Mork-2.")
    parser.add_argument(
        "--fresh",
        "--no-resume",
        action="store_true",
        help="Start from scratch (ignore checkpoints/latest.pt).",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help=(
            "Load model weights from this checkpoint, then start a new LR "
            "schedule at step 0 with a fresh optimizer. "
            "Ignored if resume finds latest.pt in --checkpoint-dir."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=f"Step to stop training at (default: {defaults.max_steps}).",
    )
    parser.add_argument(
        "--cooldown-start",
        type=int,
        default=None,
        help=(
            "Step where the LR starts its linear decay to min_lr "
            "(default: none, constant LR)."
        ),
    )
    parser.add_argument(
        "--cooldown-steps",
        type=int,
        default=None,
        help=f"Length of the linear cooldown (default: {defaults.cooldown_steps}).",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=None,
        help=f"Cooldown target as a fraction of max_lr (default: {defaults.min_lr_ratio}).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Log to runs/<run-name> (created if missing). Default: reuse the "
            "checkpoint's run on resume, else a new timestamped run."
        ),
    )
    parser.add_argument(
        "--max-lr",
        type=float,
        default=None,
        help=f"Peak learning rate (default: {defaults.max_lr}).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help=f"LR warmup steps (default: {defaults.warmup_steps}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=f"Device (default: {defaults.device}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"RNG seed (default: {defaults.seed}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=f"Checkpoint directory (default: {defaults.checkpoint_dir}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Token shard directory (default: {defaults.data_dir}).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile.",
    )
    return parser.parse_args(argv)


def configs_from_args(
    args: argparse.Namespace,
) -> tuple[TrainConfig, ModelConfig]:
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()
    overrides: dict = {}
    if args.fresh:
        overrides["resume"] = False
    if args.init_from is not None:
        overrides["init_from"] = args.init_from
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.cooldown_start is not None:
        overrides["cooldown_start"] = args.cooldown_start
    if args.cooldown_steps is not None:
        if args.cooldown_steps <= 0:
            raise SystemExit("--cooldown-steps must be positive")
        overrides["cooldown_steps"] = args.cooldown_steps
    if args.min_lr_ratio is not None:
        overrides["min_lr_ratio"] = args.min_lr_ratio
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if args.max_lr is not None:
        overrides["max_lr"] = args.max_lr
    if args.warmup_steps is not None:
        overrides["warmup_steps"] = args.warmup_steps
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.no_compile:
        overrides["compile"] = False
    if overrides:
        train_cfg = replace(train_cfg, **overrides)
    return train_cfg, model_cfg


def configure_optimizers(
    model: GPT, weight_decay: float, learning_rate: float
) -> torch.optim.AdamW:
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    unique_params = list({id(p): p for p in param_dict.values()}.values())

    decay_params = [p for p in unique_params if p.dim() >= 2]
    nodecay_params = [p for p in unique_params if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, "
        f"with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, "
        f"with {num_nodecay_params:,} parameters"
    )

    return torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True
    )


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * step / cfg.warmup_steps
    if cfg.cooldown_start is None or step < cfg.cooldown_start:
        return cfg.max_lr
    progress = min(1.0, (step - cfg.cooldown_start) / cfg.cooldown_steps)
    return cfg.max_lr - progress * (cfg.max_lr - cfg.min_lr)


def describe_schedule(cfg: TrainConfig, start_step: int) -> str:
    if cfg.cooldown_start is None:
        return (
            f"lr: constant {cfg.max_lr:.4e} after {cfg.warmup_steps} warmup steps, "
            f"no cooldown; stops at step {cfg.max_steps:,}"
        )
    end = cfg.cooldown_start + cfg.cooldown_steps
    text = (
        f"lr: constant {cfg.max_lr:.4e} until step {cfg.cooldown_start:,}, "
        f"then linear to {cfg.min_lr:.4e} by step {end:,}; stops at step {cfg.max_steps:,}"
    )
    if cfg.max_steps < end:
        text += (
            f"\nwarning: max_steps ends the run before the cooldown finishes; "
            f"the last lr will be {get_lr(cfg.max_steps - 1, cfg):.4e}"
        )
    if start_step > cfg.cooldown_start:
        text += f"\nnote: resuming inside the cooldown at lr {get_lr(start_step, cfg):.4e}"
    return text


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    loader: DataLoaderLite,
    *,
    device: str,
    dtype: torch.dtype,
    num_batches: int,
) -> float:
    """Mean loss on the first ``num_batches`` of ``loader``: the same tokens every call."""
    loader.reset()
    model.eval()
    losses = torch.zeros(num_batches, device=device)
    for i in range(num_batches):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(x, y)
        assert loss is not None
        losses[i] = loss
    model.train()
    return float(losses.mean())


def main(
    argv: list[str] | None = None,
    train_cfg: TrainConfig | None = None,
    model_cfg: ModelConfig | None = None,
    train_loader: DataLoaderLite | None = None,
    val_loader: DataLoaderLite | None = None,
) -> None:
    # The console script calls main() with no configs, so flags have to be read here.
    if train_cfg is None and model_cfg is None:
        train_cfg, model_cfg = configs_from_args(parse_args(argv))
    else:
        train_cfg = train_cfg or TrainConfig()
        model_cfg = model_cfg or ModelConfig()

    if train_cfg.seq_len > model_cfg.block_size:
        raise ValueError(
            f"seq_len ({train_cfg.seq_len}) must be <= block_size ({model_cfg.block_size})"
        )

    device = train_cfg.device
    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(train_cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed(train_cfg.seed)

    print(f"total desired batch size: {train_cfg.total_batch_size:,}")
    print(f"=> calculated gradient accumulation steps: {train_cfg.grad_accum_steps:,}")

    if (train_loader is None) != (val_loader is None):
        raise ValueError("pass both loaders or neither")
    if train_loader is None or val_loader is None:
        train_loader = DataLoaderLite(
            B=train_cfg.batch_size,
            T=train_cfg.seq_len,
            split="train",
            data_dir=train_cfg.data_dir,
            shuffle_chunk_tokens=train_cfg.shuffle_chunk_tokens,
        )
        val_loader = DataLoaderLite(
            B=train_cfg.batch_size,
            T=train_cfg.seq_len,
            split="val",
            data_dir=train_cfg.data_dir,
            shuffle_chunk_tokens=train_cfg.shuffle_chunk_tokens,
            fixed_shuffle=True,
            shuffle_seed=train_cfg.seed,
        )

    torch.set_float32_matmul_precision("high")

    raw_model = GPT(model_cfg).to(device)
    optimizer = configure_optimizers(
        raw_model, weight_decay=train_cfg.weight_decay, learning_rate=train_cfg.max_lr
    )

    start_step = 0
    saved_batch: int | None = None
    checkpoint_run_dir: str | None = None
    latest_path = latest_checkpoint_path(train_cfg.checkpoint_dir)
    init_from = train_cfg.init_from

    if train_cfg.resume and latest_path is not None:
        loaded_step, loaded_model_cfg, loaded_loss, checkpoint_run_dir, saved_batch = load_checkpoint(
            latest_path,
            raw_model,
            optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        if loaded_model_cfg != model_cfg:
            print("warning: checkpoint model config differs from current ModelConfig")
        start_step = loaded_step + 1
        print(
            f"resumed from {latest_path} "
            f"(completed step {loaded_step}, loss {loaded_loss:.6f}) "
            f"-> starting at step {start_step}"
        )
        if init_from is not None:
            print(f"note: ignoring --init-from {init_from} because resume found {latest_path}")
    elif init_from is not None:
        if not init_from.exists():
            raise SystemExit(f"--init-from not found: {init_from}")
        loaded_model_cfg, src_step, src_loss = load_weights_only(init_from, raw_model)
        if loaded_model_cfg != model_cfg:
            print(
                "warning: --init-from model config differs from current ModelConfig; "
                "weights loaded anyway"
            )
        # Fresh optimizer after weight load (Adam moments must not carry over).
        optimizer = configure_optimizers(
            raw_model,
            weight_decay=train_cfg.weight_decay,
            learning_rate=train_cfg.max_lr,
        )
        start_step = 0
        print(
            f"init from {init_from} "
            f"(source step {src_step}, loss {src_loss:.6f}) "
            f"-> new schedule from step 0, max_lr={train_cfg.max_lr:.4e}, "
            f"max_steps={train_cfg.max_steps}"
        )
        if latest_path is not None and not train_cfg.resume:
            print(
                f"note: {latest_path} exists but resume=False; "
                f"initialized from --init-from instead"
            )
    elif latest_path is not None:
        print(f"found {latest_path} but resume=False, training from scratch")

    best_val_loss = float("inf")
    best_val_info = load_best_val_loss(train_cfg.checkpoint_dir, val_slice_tag(train_cfg))
    resuming = train_cfg.resume and start_step > 0
    if resuming and best_val_info is None and best_val_checkpoint_path(train_cfg.checkpoint_dir):
        print(
            "best_val.pt was scored on a different val slice; "
            "the first eval of this run sets a new best"
        )
    if best_val_info is not None and resuming:
        best_val_loss, best_val_step = best_val_info
        print(
            f"tracking best val from checkpoint: "
            f"val_loss {best_val_loss:.6f} (step {best_val_step})"
        )

    print(describe_schedule(train_cfg, start_step))

    run_dir = resolve_run_dir(
        train_cfg,
        resume=train_cfg.resume and start_step > 0,
        checkpoint_run_dir=checkpoint_run_dir,
    )
    metrics = MetricsLogger(run_dir)
    print(f"logging metrics to {metrics.metrics_path}")

    model: GPT | torch.nn.Module = (
        torch.compile(raw_model) if train_cfg.compile else raw_model
    )

    dtype = torch.bfloat16 if train_cfg.dtype == "bfloat16" else torch.float16

    if saved_batch is not None and saved_batch != train_cfg.batch_size:
        _set_micro_batch(train_cfg, train_loader, val_loader, int(saved_batch))
        print(f"restored micro-batch {saved_batch} from checkpoint")

    if train_cfg.auto_batch_probe and start_step == 0:
        optimizer = probe_auto_batch(
            model=model,
            raw_model=raw_model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=train_cfg,
            device=device,
            dtype=dtype,
            max_dt_s=train_cfg.auto_batch_max_dt_s,
        )

    if start_step == 0:
        # Probe steps read ahead. Rewind so step 0 starts at the first batch.
        train_loader.reset()
        val_loader.reset()

    # Write configs after probe so batch_size / grad_accum reflect the choice.
    metrics.write_configs(model_cfg, train_cfg)
    print(
        f"using batch_size={train_cfg.batch_size}, "
        f"grad_accum={train_cfg.grad_accum_steps}, "
        f"tokens/step={train_cfg.total_batch_size:,}"
    )

    last_completed_step = start_step - 1
    last_loss = float("nan")
    run_t0 = time.time()

    try:
        for step in range(start_step, train_cfg.max_steps):
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)

            loss_accum = 0.0
            for _ in range(train_cfg.grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)

                with torch.autocast(device_type=device, dtype=dtype):
                    _, loss = model(x, y)
                assert loss is not None
                loss = loss / train_cfg.grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), train_cfg.grad_clip
            )

            lr = get_lr(step, train_cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000
            elapsed_s = t1 - run_t0
            tokens_processed = (
                train_loader.B * train_loader.T * train_cfg.grad_accum_steps
            )
            tokens_per_sec = tokens_processed / (t1 - t0)
            print(
                f"step {step:4d} | loss {loss_accum:.6f} | lr {lr:.4e} | "
                f"norm {norm:.4f} | dt {dt:.2f}ms | tokens/sec {tokens_per_sec:.2f} | "
                f"elapsed {_format_elapsed(elapsed_s)}"
            )

            last_completed_step = step
            last_loss = float(loss_accum)
            metrics.log(
                step=step,
                split="train",
                loss=last_loss,
                lr=lr,
                grad_norm=float(norm),
                dt_ms=dt,
                tokens_per_sec=tokens_per_sec,
                elapsed_s=elapsed_s,
            )

            completed = step + 1
            if completed % train_cfg.eval_every == 0 or completed == train_cfg.max_steps:
                val_loss = estimate_loss(
                    model,
                    val_loader,
                    device=device,
                    dtype=dtype,
                    num_batches=train_cfg.eval_batches,
                )
                elapsed_s = time.time() - run_t0
                print(
                    f"step {step:4d} | val_loss {val_loss:.6f} "
                    f"(fixed {train_cfg.eval_tokens:,} tokens) | "
                    f"elapsed {_format_elapsed(elapsed_s)}"
                )
                metrics.log(
                    step=step,
                    split="val",
                    loss=val_loss,
                    elapsed_s=elapsed_s,
                )
                best_val_loss = maybe_save_best_val(
                    step=step,
                    train_loss=last_loss,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    raw_model=raw_model,
                    optimizer=optimizer,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    run_dir=run_dir,
                )

            maybe_save_checkpoints(
                step,
                last_loss,
                raw_model,
                optimizer,
                model_cfg,
                train_cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                run_dir=run_dir,
            )

    except KeyboardInterrupt:
        if last_completed_step >= 0 and train_cfg.save_checkpoints:
            print(f"\ninterrupted after step {last_completed_step}, saving checkpoint...")
            save_checkpoint(
                train_cfg.checkpoint_dir / "latest.pt",
                step=last_completed_step,
                loss=last_loss,
                raw_model=raw_model,
                optimizer=optimizer,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                run_dir=run_dir,
            )
            print(f"saved checkpoint: {train_cfg.checkpoint_dir / 'latest.pt'}")
        elif last_completed_step >= 0:
            print(
                f"\ninterrupted after step {last_completed_step} "
                "(save_checkpoints=False; nothing written)"
            )
        raise
    finally:
        metrics.close()

    if last_completed_step >= 0 and train_cfg.save_checkpoints:
        save_checkpoint(
            train_cfg.checkpoint_dir / "latest.pt",
            step=last_completed_step,
            loss=last_loss,
            raw_model=raw_model,
            optimizer=optimizer,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            run_dir=run_dir,
        )
        print(f"training complete, saved checkpoint: {train_cfg.checkpoint_dir / 'latest.pt'}")
    elif last_completed_step >= 0:
        print("training complete (save_checkpoints=False; no checkpoint written)")


if __name__ == "__main__":
    main()

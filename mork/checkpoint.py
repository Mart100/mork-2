from dataclasses import asdict
from pathlib import Path

import torch

from mork.config import ModelConfig, TrainConfig
from mork.data import DataLoaderLite
from mork.model import GPT


def _train_config_to_dict(cfg: TrainConfig) -> dict:
    data = asdict(cfg)
    data["data_dir"] = str(data["data_dir"])
    data["checkpoint_dir"] = str(data["checkpoint_dir"])
    data["runs_dir"] = str(data["runs_dir"])
    init_from = data.get("init_from")
    data["init_from"] = str(init_from) if init_from is not None else None
    return data


def val_slice_tag(cfg: TrainConfig) -> str:
    """Identifies the val tokens a val_loss was measured on; losses only compare within a tag."""
    return (
        f"fixed:{cfg.eval_tokens}:seq{cfg.seq_len}:"
        f"chunk{cfg.shuffle_chunk_tokens}:seed{cfg.seed}"
    )


def latest_checkpoint_path(checkpoint_dir: Path) -> Path | None:
    path = checkpoint_dir / "latest.pt"
    return path if path.exists() else None


def best_val_checkpoint_path(checkpoint_dir: Path) -> Path | None:
    path = checkpoint_dir / "best_val.pt"
    return path if path.exists() else None


def save_checkpoint(
    path: Path,
    *,
    step: int,
    loss: float,
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_loader: DataLoaderLite | None = None,
    val_loader: DataLoaderLite | None = None,
    run_dir: Path | None = None,
    val_loss: float | None = None,
    val_slice: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "loss": loss,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_cfg),
        "train_config": _train_config_to_dict(train_cfg),
    }
    if val_loss is not None:
        payload["val_loss"] = val_loss
    if val_slice is not None:
        payload["val_slice"] = val_slice
    if train_loader is not None:
        payload["train_loader"] = train_loader.state_dict()
    if val_loader is not None:
        payload["val_loader"] = val_loader.state_dict()
    if run_dir is not None:
        payload["run_dir"] = str(run_dir)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: Path,
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoaderLite | None = None,
    val_loader: DataLoaderLite | None = None,
) -> tuple[int, ModelConfig, float, str | None, int | None]:
    device = next(raw_model.parameters()).device
    ckpt = torch.load(path, map_location=device, weights_only=False)

    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if train_loader is not None and "train_loader" in ckpt:
        train_loader.load_state_dict(ckpt["train_loader"])
    if val_loader is not None and "val_loader" in ckpt:
        val_loader.load_state_dict(ckpt["val_loader"])

    step = int(ckpt["step"])
    model_cfg = ModelConfig(**ckpt["model_config"])
    loss = float(ckpt.get("loss", float("nan")))
    run_dir = ckpt.get("run_dir")
    saved_batch = ckpt.get("train_config", {}).get("batch_size")
    return step, model_cfg, loss, run_dir, saved_batch


def load_weights_only(
    path: Path,
    raw_model: GPT,
) -> tuple[ModelConfig, int, float]:
    """Load model weights only. Skips optimizer, loaders, and step."""
    device = next(raw_model.parameters()).device
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])
    model_cfg = ModelConfig(**ckpt["model_config"])
    step = int(ckpt["step"])
    loss = float(ckpt.get("val_loss", ckpt.get("loss", float("nan"))))
    return model_cfg, step, loss


def load_model_for_inference(
    path: Path, device: str = "cuda"
) -> tuple[GPT, ModelConfig, int, float]:
    # mmap on CPU so the optimizer state is never read into RAM or VRAM.
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    with torch.device(device):
        model = GPT(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = int(ckpt["step"])
    loss = float(ckpt.get("val_loss", ckpt.get("loss", float("nan"))))
    return model, model_cfg, step, loss


def load_best_val_loss(checkpoint_dir: Path, val_slice: str) -> tuple[float, int] | None:
    """Best val loss so far, or None if best_val.pt was scored on another slice."""
    path = best_val_checkpoint_path(checkpoint_dir)
    if path is None:
        return None
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    val_loss = ckpt.get("val_loss")
    if val_loss is None or ckpt.get("val_slice") != val_slice:
        return None
    return float(val_loss), int(ckpt["step"])


def maybe_save_best_val(
    *,
    step: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_loader: DataLoaderLite | None = None,
    val_loader: DataLoaderLite | None = None,
    run_dir: Path | None = None,
) -> float:
    """Save ``best_val.pt`` when ``val_loss`` improves. Returns updated best."""
    if not train_cfg.save_checkpoints:
        return min(best_val_loss, val_loss)
    if val_loss >= best_val_loss:
        return best_val_loss

    path = train_cfg.checkpoint_dir / "best_val.pt"
    save_checkpoint(
        path,
        step=step,
        loss=train_loss,
        raw_model=raw_model,
        optimizer=optimizer,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
        val_loss=val_loss,
        val_slice=val_slice_tag(train_cfg),
    )
    print(
        f"saved best-val checkpoint: {path} "
        f"(step {step}, val_loss {val_loss:.6f}, prev best {best_val_loss:.6f})"
    )
    return val_loss


def maybe_save_checkpoints(
    step: int,
    loss: float,
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_loader: DataLoaderLite | None = None,
    val_loader: DataLoaderLite | None = None,
    run_dir: Path | None = None,
) -> None:
    if not train_cfg.save_checkpoints:
        return

    completed = step + 1
    checkpoint_dir = train_cfg.checkpoint_dir

    if completed % train_cfg.checkpoint_every != 0:
        return

    latest_path = checkpoint_dir / "latest.pt"
    save_checkpoint(
        latest_path,
        step=step,
        loss=loss,
        raw_model=raw_model,
        optimizer=optimizer,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
    )
    print(f"saved checkpoint: {latest_path} (step {step}, loss {loss:.6f})")

    if completed % train_cfg.snapshot_every == 0:
        snapshot_path = checkpoint_dir / f"step_{completed:06d}.pt"
        save_checkpoint(
            snapshot_path,
            step=step,
            loss=loss,
            raw_model=raw_model,
            optimizer=optimizer,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            run_dir=run_dir,
        )
        print(f"saved snapshot: {snapshot_path}")

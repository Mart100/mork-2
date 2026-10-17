import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from mork.config import ModelConfig, TrainConfig


METRICS_FIELDS = [
    "step",
    "split",
    "loss",
    "lr",
    "grad_norm",
    "dt_ms",
    "tokens_per_sec",
    "elapsed_s",
    "time",
]


def _config_to_jsonable(cfg: ModelConfig | TrainConfig) -> dict:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def create_run_dir(runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class MetricsLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self._file = open(self.metrics_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=METRICS_FIELDS)
        if self.metrics_path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def write_configs(self, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
        payload = {
            "model": _config_to_jsonable(model_cfg),
            "train": _config_to_jsonable(train_cfg),
        }
        path = self.run_dir / "config.json"
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log(
        self,
        *,
        step: int,
        split: str,
        loss: float,
        lr: float | None = None,
        grad_norm: float | None = None,
        dt_ms: float | None = None,
        tokens_per_sec: float | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        self._writer.writerow(
            {
                "step": step,
                "split": split,
                "loss": f"{loss:.6f}",
                "lr": "" if lr is None else f"{lr:.8e}",
                "grad_norm": "" if grad_norm is None else f"{grad_norm:.6f}",
                "dt_ms": "" if dt_ms is None else f"{dt_ms:.2f}",
                "tokens_per_sec": (
                    "" if tokens_per_sec is None else f"{tokens_per_sec:.2f}"
                ),
                "elapsed_s": "" if elapsed_s is None else f"{elapsed_s:.2f}",
                "time": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def resolve_run_dir(
    train_cfg: TrainConfig,
    *,
    resume: bool,
    checkpoint_run_dir: str | None = None,
) -> Path:
    """Reuse checkpoint run dir on resume; otherwise create a new run folder."""
    if resume and checkpoint_run_dir:
        run_dir = Path(checkpoint_run_dir)
        if run_dir.exists():
            return run_dir
        print(f"warning: checkpoint run_dir {run_dir} missing, creating a new run")

    return create_run_dir(train_cfg.runs_dir)

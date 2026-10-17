"""Build the pretrain corpus from HuggingFaceTB/smollm-corpus.

Text budget (train and val together):

- 3 GB of high-scoring Python-Edu (source fetched from Software Heritage)
- 3 GB of Cosmopedia v2
- 2 GB of FineWeb-Edu dedup

One percent of each source is held out for validation. A byte-level BPE is
trained on a sample of the training text, then documents are interleaved in
the 3:3:2 mix and written as ``corpus_{split}_XXXX.npy`` shards.
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset

from mork.config import PROJECT_ROOT, ModelConfig
from mork.tokenizer import DEFAULT_VOCAB_SIZE, train_tokenizer_from_iterator

DATASET = "HuggingFaceTB/smollm-corpus"
GB = 1_000_000_000
PYTHON_BYTES = 3 * GB
COSMOPEDIA_BYTES = 3 * GB
FINEWEB_BYTES = 2 * GB
VAL_RATIO = 0.01
SEED = 1337
SHARD_TOKENS = 20_000_000
TOKENIZER_SAMPLE_BYTES = 128 * 1024 * 1024
MIN_PYTHON_BYTES = 4_096
MAX_PYTHON_BYTES = 100_000
MAX_TEXT_CHARS = 500_000
MIN_TEXT_CHARS = 200
SHUFFLE_BUFFER = 512
ROUND_UNIT_BYTES = 1_000_000
ENCODE_BATCH = 64
DOWNLOAD_WORKERS = 64

CANDIDATE_DTYPE = np.dtype(
    [
        ("score", "<f4"),
        ("length", "<i4"),
        ("blob", "S20"),
    ]
)

BLOB_URLS = (
    "https://softwareheritage.s3.amazonaws.com/content/{blob_id}",
    "https://softwareheritage.s3.us-west-2.amazonaws.com/content/{blob_id}",
)

SOURCES = (
    ("python", "python-edu", PYTHON_BYTES),
    ("cosmopedia", "cosmopedia-v2", COSMOPEDIA_BYTES),
    ("fineweb", "fineweb-edu-dedup", FINEWEB_BYTES),
)


@dataclass
class PrepareConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    python_bytes: int = PYTHON_BYTES
    cosmopedia_bytes: int = COSMOPEDIA_BYTES
    fineweb_bytes: int = FINEWEB_BYTES
    val_ratio: float = VAL_RATIO
    seed: int = SEED
    workers: int = DOWNLOAD_WORKERS
    shard_tokens: int = SHARD_TOKENS
    vocab_size: int = DEFAULT_VOCAB_SIZE
    tokenizer_sample_bytes: int = TOKENIZER_SAMPLE_BYTES
    force_encode: bool = False

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def budget(self, source: str) -> int:
        return {
            "python": self.python_bytes,
            "cosmopedia": self.cosmopedia_bytes,
            "fineweb": self.fineweb_bytes,
        }[source]


def _split_targets(total: int, val_ratio: float) -> tuple[int, int]:
    val = max(1, int(total * val_ratio))
    return total - val, val


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _fmt_bytes(n: int) -> str:
    if n >= GB:
        return f"{n / GB:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    return f"{n} B"


def _append_doc(handle, text: str) -> int:
    line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
    handle.write(line)
    return len(text.encode("utf-8"))


def _truncate(path: Path, pos: int) -> None:
    if not path.exists():
        return
    with path.open("r+b") as handle:
        handle.truncate(pos)


class JsonlSplit:
    """Append-only JSONL plus a meta file so a crashed download can resume."""

    def __init__(self, path: Path):
        self.path = path
        self.meta_path = path.with_suffix(".meta.json")
        self.meta = _read_json(self.meta_path)
        self.docs = int(self.meta.get("docs", 0))
        self.text_bytes = int(self.meta.get("text_bytes", 0))
        self.file_pos = int(self.meta.get("file_pos", 0))
        self.complete = bool(self.meta.get("complete", False))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > self.file_pos:
            _truncate(path, self.file_pos)
        elif path.exists() and path.stat().st_size < self.file_pos:
            self.docs = 0
            self.text_bytes = 0
            self.file_pos = 0
            _truncate(path, 0)
        self._handle = path.open("a", encoding="utf-8")

    def add(self, text: str) -> None:
        self.text_bytes += _append_doc(self._handle, text)
        self.docs += 1

    def checkpoint(self, extra: dict | None = None) -> None:
        self._handle.flush()
        self.file_pos = self.path.stat().st_size
        payload = dict(self.meta)
        payload.update(
            {
                "docs": self.docs,
                "text_bytes": self.text_bytes,
                "file_pos": self.file_pos,
                "complete": self.complete,
            }
        )
        if extra:
            payload.update(extra)
        self.meta = payload
        _write_json(self.meta_path, payload)

    def finish(self, extra: dict | None = None) -> None:
        self.complete = True
        self.checkpoint(extra)
        self._handle.close()

    def close(self) -> None:
        if not self._handle.closed:
            self.checkpoint()
            self._handle.close()


def _stream(config_name: str):
    dataset = load_dataset(DATASET, config_name, split="train", streaming=True)
    dataset = dataset.select_columns(["text"])
    # One parquet shard at a time. The default of 10 holds several multi-gigabyte
    # files in memory at once, which the kernel killed on this machine.
    return dataset.shuffle(
        seed=SEED,
        buffer_size=SHUFFLE_BUFFER,
        max_buffer_input_shards=1,
    )


def collect_python_candidates(path: Path) -> None:
    """Write one packed record per Python file in the size window."""
    if path.exists() and path.stat().st_size > 0 and path.with_suffix(".done").exists():
        print(f"python candidates ready: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print("scanning python-edu metadata...", flush=True)
    dataset = load_dataset(DATASET, "python-edu", split="train", streaming=True)
    dataset = dataset.select_columns(["blob_id", "path", "length_bytes", "score"])
    kept = 0
    seen = 0
    t0 = time.time()
    with path.open("wb") as handle:
        for row in dataset:
            seen += 1
            length = int(row["length_bytes"] or 0)
            name = row["path"] or ""
            blob_id = row["blob_id"] or ""
            if not (MIN_PYTHON_BYTES <= length <= MAX_PYTHON_BYTES):
                continue
            if not name.endswith(".py") or len(blob_id) != 40:
                continue
            try:
                blob = bytes.fromhex(blob_id)
            except ValueError:
                continue
            record = np.zeros(1, dtype=CANDIDATE_DTYPE)
            record["score"] = float(row["score"])
            record["length"] = length
            record["blob"] = blob
            handle.write(record.tobytes())
            kept += 1
            if seen % 200_000 == 0:
                print(
                    f"  scanned {seen:,} rows, kept {kept:,} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
    path.with_suffix(".done").write_text(f"{kept}\n", encoding="utf-8")
    print(f"kept {kept:,} python candidates from {seen:,} rows", flush=True)


def _load_candidates(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=CANDIDATE_DTYPE)
    if len(data) == 0:
        raise RuntimeError(f"no python candidates in {path}")
    order = np.lexsort((-data["score"], data["blob"]))
    data = data[order]
    blobs = data["blob"]
    keep = np.empty(len(data), dtype=bool)
    keep[0] = True
    keep[1:] = blobs[1:] != blobs[:-1]
    data = data[keep]
    return data[np.argsort(-data["score"], kind="mergesort")]


def _plan_python(
    candidates: np.ndarray,
    train_path: Path,
    val_path: Path,
    train_target: int,
    val_target: int,
    seed: int,
    val_ratio: float,
) -> None:
    """Highest scores first, skipping tiny files, with extras for failed downloads."""
    candidates = candidates[candidates["length"] >= MIN_PYTHON_BYTES]
    rng = np.random.default_rng(seed)
    train_bytes = 0
    val_bytes = 0
    train_cap = int(train_target * 1.25)
    val_cap = int(val_target * 1.25)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_count = 0
    val_count = 0
    with train_path.open("wb") as train_out, val_path.open("wb") as val_out:
        for row in candidates:
            if train_bytes >= train_cap and val_bytes >= val_cap:
                break
            length = int(row["length"])
            blob = np.asarray(row["blob"], dtype="S20").tobytes()
            need_train = train_bytes < train_cap
            need_val = val_bytes < val_cap
            if need_train and need_val:
                to_val = bool(rng.random() < val_ratio)
            else:
                to_val = need_val
            if to_val and need_val:
                val_out.write(blob)
                val_bytes += length
                val_count += 1
            elif need_train:
                train_out.write(blob)
                train_bytes += length
                train_count += 1
    print(
        f"python plan: train {train_count:,} files (~{_fmt_bytes(train_bytes)}), "
        f"val {val_count:,} files (~{_fmt_bytes(val_bytes)})",
        flush=True,
    )


_connections = threading.local()


def _fetch_url(url: str) -> bytes:
    """GET one blob, reusing this thread's TLS connection to the same host.

    A fresh handshake to the Software Heritage bucket costs more than the
    transfer of these small files. Worker threads keep one connection open.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname is None:
        raise urllib.error.URLError(f"unsupported url {url}")
    key = (parsed.hostname, parsed.port or 443)
    conn = getattr(_connections, "conn", None)
    if conn is None or getattr(_connections, "key", None) != key:
        if conn is not None:
            conn.close()
        conn = http.client.HTTPSConnection(parsed.hostname, key[1], timeout=60)
        _connections.conn = conn
        _connections.key = key
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    try:
        conn.request(
            "GET",
            path,
            headers={"User-Agent": "mork-2-prepare", "Connection": "keep-alive"},
        )
        response = conn.getresponse()
        payload = response.read()
    except Exception:
        conn.close()
        _connections.conn = None
        raise
    if response.status in {301, 302, 303, 307, 308}:
        location = response.getheader("Location")
        if not location:
            raise urllib.error.URLError(f"redirect without location from {url}")
        return _fetch_url(urllib.parse.urljoin(url, location))
    if response.status != 200:
        raise urllib.error.HTTPError(url, response.status, response.reason, response.headers, None)
    return payload


def _resolve_blob_url(blob_ids: list[str]) -> str:
    last_error: Exception | None = None
    for blob_id in blob_ids[:8]:
        for template in BLOB_URLS:
            url = template.format(blob_id=blob_id)
            try:
                payload = _fetch_url(url)
            except Exception as exc:
                last_error = exc
                continue
            if payload[:2] == b"\x1f\x8b":
                return template
    raise RuntimeError("could not download a sample Python blob") from last_error


def _download_blob(blob_id: str, url_template: str) -> str | None:
    url = url_template.format(blob_id=blob_id)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            payload = _fetch_url(url)
            break
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 404}:
                return None
            last_error = exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            ssl.SSLError,
            http.client.HTTPException,
            gzip.BadGzipFile,
        ) as exc:
            last_error = exc
        time.sleep(0.5 * (attempt + 1))
    else:
        print(f"  skip {blob_id}: {last_error}", flush=True)
        return None
    if payload[:2] == b"\x1f\x8b":
        try:
            payload = gzip.decompress(payload)
        except gzip.BadGzipFile:
            return None
    if len(payload) > MAX_PYTHON_BYTES * 2 or b"\x00" in payload:
        return None
    text = payload.decode("utf-8", errors="ignore").strip()
    if len(text) < MIN_TEXT_CHARS:
        return None
    return text[:MAX_TEXT_CHARS]


def _iter_plan(path: Path, start: int):
    blobs = np.memmap(path, dtype="S20", mode="r")
    for index in range(start, len(blobs)):
        yield index, blobs[index].tobytes().hex()


def download_python_split(
    plan_path: Path,
    dest: JsonlSplit,
    target_bytes: int,
    url_template: str,
    workers: int,
) -> None:
    if dest.complete and dest.text_bytes >= target_bytes:
        print(f"{dest.path.name} already {_fmt_bytes(dest.text_bytes)}", flush=True)
        return
    start = int(dest.meta.get("plan_index", 0))
    plan_size = plan_path.stat().st_size // 20
    print(
        f"downloading {dest.path.name}: {_fmt_bytes(dest.text_bytes)} / "
        f"{_fmt_bytes(target_bytes)}, plan {start:,}/{plan_size:,}",
        flush=True,
    )
    finished = start
    done_out_of_order: set[int] = set()
    blob_iter = _iter_plan(plan_path, start)
    stop = dest.text_bytes >= target_bytes
    t0 = time.time()
    fetched = 0

    def note_finished(index: int) -> None:
        nonlocal finished
        done_out_of_order.add(index)
        while finished in done_out_of_order:
            done_out_of_order.remove(finished)
            finished += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        inflight: dict = {}

        def submit_one() -> bool:
            try:
                index, blob_id = next(blob_iter)
            except StopIteration:
                return False
            inflight[pool.submit(_download_blob, blob_id, url_template)] = index
            return True

        for _ in range(max(1, workers * 2)):
            if not submit_one():
                break
        while inflight:
            future = next(as_completed(list(inflight)))
            index = inflight.pop(future)
            text = future.result()
            fetched += 1
            note_finished(index)
            if text and not stop:
                dest.add(text)
                if dest.text_bytes >= target_bytes:
                    stop = True
            if fetched % 200 == 0:
                dest.checkpoint({"plan_index": finished})
            if fetched % 500 == 0:
                rate = fetched / max(time.time() - t0, 1e-6)
                print(
                    f"  {dest.path.name} {_fmt_bytes(dest.text_bytes)} / "
                    f"{_fmt_bytes(target_bytes)}  docs={dest.docs:,}  "
                    f"{rate:.1f} files/s",
                    flush=True,
                )
            if not stop:
                submit_one()
    dest.checkpoint({"plan_index": finished})
    if dest.text_bytes < int(target_bytes * 0.9):
        raise RuntimeError(
            f"{dest.path.name} reached {_fmt_bytes(dest.text_bytes)}, "
            f"short of {_fmt_bytes(target_bytes)}"
        )
    dest.finish()


def download_python(cfg: PrepareConfig) -> None:
    raw = cfg.raw_dir
    candidate_path = raw / "python_candidates.bin"
    collect_python_candidates(candidate_path)
    train_target, val_target = _split_targets(cfg.python_bytes, cfg.val_ratio)
    train_plan = raw / "python_train.plan"
    val_plan = raw / "python_val.plan"
    if not (train_plan.exists() and val_plan.exists()):
        _plan_python(
            _load_candidates(candidate_path),
            train_plan,
            val_plan,
            train_target,
            val_target,
            cfg.seed,
            cfg.val_ratio,
        )
    if train_plan.stat().st_size < 20:
        raise RuntimeError("python plan is empty")
    sample_ids = []
    for index, blob_id in _iter_plan(train_plan, 0):
        sample_ids.append(blob_id)
        if index >= 7:
            break
    url_template = _resolve_blob_url(sample_ids)
    print(f"software heritage url: {url_template}", flush=True)
    train = JsonlSplit(raw / "python_train.jsonl")
    val = JsonlSplit(raw / "python_val.jsonl")
    try:
        download_python_split(train_plan, train, train_target, url_template, cfg.workers)
        download_python_split(
            val_plan, val, val_target, url_template, max(4, cfg.workers // 4)
        )
    finally:
        train.close()
        val.close()


def stream_text_source(
    name: str,
    config_name: str,
    total_bytes: int,
    val_ratio: float,
    raw_dir: Path,
) -> None:
    train_target, val_target = _split_targets(total_bytes, val_ratio)
    train = JsonlSplit(raw_dir / f"{name}_train.jsonl")
    val = JsonlSplit(raw_dir / f"{name}_val.jsonl")
    if train.complete and val.complete:
        print(
            f"{name} ready: train {_fmt_bytes(train.text_bytes)}, "
            f"val {_fmt_bytes(val.text_bytes)}",
            flush=True,
        )
        train.close()
        val.close()
        return
    next_index = int(train.meta.get("next_index", 0))
    print(
        f"streaming {name} from row {next_index:,}: "
        f"train {_fmt_bytes(train.text_bytes)}/{_fmt_bytes(train_target)}, "
        f"val {_fmt_bytes(val.text_bytes)}/{_fmt_bytes(val_target)}",
        flush=True,
    )
    dataset = _stream(config_name)
    if next_index:
        dataset = dataset.skip(next_index)
    seen = next_index
    t0 = time.time()
    try:
        for row in dataset:
            seen += 1
            if train.text_bytes >= train_target and val.text_bytes >= val_target:
                break
            text = (row.get("text") or "").strip()
            if not (MIN_TEXT_CHARS <= len(text) <= MAX_TEXT_CHARS):
                continue
            digest = zlib.crc32(text[:256].encode("utf-8")) % 1000
            to_val = digest < int(val_ratio * 1000)
            if to_val and val.text_bytes < val_target:
                val.add(text)
            elif not to_val and train.text_bytes < train_target:
                train.add(text)
            if seen % 500 == 0:
                train.checkpoint({"next_index": seen})
                val.checkpoint({"next_index": seen})
                elapsed = max(time.time() - t0, 1e-6)
                gained = train.text_bytes + val.text_bytes
                print(
                    f"  {name} train {_fmt_bytes(train.text_bytes)} "
                    f"val {_fmt_bytes(val.text_bytes)}  "
                    f"rows={seen:,}  {gained / elapsed / 1e6:.2f} MB/s",
                    flush=True,
                )
    finally:
        train.checkpoint({"next_index": seen})
        val.checkpoint({"next_index": seen})
    if train.text_bytes < train_target * 0.9 or val.text_bytes < val_target * 0.9:
        train.close()
        val.close()
        raise RuntimeError(
            f"{name} stopped short: train {_fmt_bytes(train.text_bytes)} "
            f"(want {_fmt_bytes(train_target)}), "
            f"val {_fmt_bytes(val.text_bytes)} (want {_fmt_bytes(val_target)})"
        )
    train.finish({"next_index": seen})
    val.finish({"next_index": seen})
    print(
        f"{name} done: train {_fmt_bytes(train.text_bytes)} "
        f"({train.docs:,} docs), val {_fmt_bytes(val.text_bytes)} "
        f"({val.docs:,} docs)",
        flush=True,
    )


def _sample_texts(path: Path, max_bytes: int, stride: int = 25) -> list[str]:
    texts: list[str] = []
    got = 0
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i % stride != 0:
                continue
            text = json.loads(line)["text"]
            texts.append(text)
            got += len(text)
            if got >= max_bytes:
                break
    return texts


def build_tokenizer(cfg: PrepareConfig) -> None:
    path = cfg.data_dir / "tokenizer.json"
    if path.exists() and not cfg.force_encode:
        from mork.tokenizer import MorkTokenizer

        existing = MorkTokenizer.load(path)
        if existing.vocab_size != cfg.vocab_size:
            raise RuntimeError(
                f"{path} has vocab {existing.vocab_size}, expected {cfg.vocab_size}"
            )
        print(f"tokenizer exists: {path}", flush=True)
        return
    weights = {
        "python": cfg.python_bytes,
        "cosmopedia": cfg.cosmopedia_bytes,
        "fineweb": cfg.fineweb_bytes,
    }
    total = sum(weights.values())
    texts: list[str] = []
    for name, weight in weights.items():
        quota = int(cfg.tokenizer_sample_bytes * weight / total)
        sample = _sample_texts(cfg.raw_dir / f"{name}_train.jsonl", quota)
        print(f"tokenizer sample {name}: {len(sample):,} docs, quota {_fmt_bytes(quota)}", flush=True)
        texts.extend(sample)
    if not texts:
        raise RuntimeError("tokenizer sample is empty")
    print(f"training byte-level BPE vocab={cfg.vocab_size:,} on {len(texts):,} docs", flush=True)
    tokenizer = train_tokenizer_from_iterator(
        texts, vocab_size=cfg.vocab_size, length=len(texts)
    )
    if tokenizer.vocab_size != cfg.vocab_size:
        raise RuntimeError(
            f"tokenizer vocab {tokenizer.vocab_size} != requested {cfg.vocab_size}"
        )
    if cfg.vocab_size == DEFAULT_VOCAB_SIZE and cfg.vocab_size != ModelConfig.vocab_size:
        raise RuntimeError(
            f"vocab {cfg.vocab_size} does not match ModelConfig.vocab_size "
            f"{ModelConfig.vocab_size}"
        )
    tokenizer.save(path)
    print(f"saved tokenizer: {path}", flush=True)


class ShardWriter:
    def __init__(self, data_dir: Path, split: str, shard_tokens: int):
        self.data_dir = data_dir
        self.split = split
        self.shard_tokens = shard_tokens
        self.parts: list[np.ndarray] = []
        self.count = 0
        self.shard_index = 0
        self.total = 0

    def add(self, ids: list[int]) -> None:
        if not ids:
            return
        array = np.asarray(ids, dtype=np.uint16)
        self.parts.append(array)
        self.count += len(array)
        self.total += len(array)
        if self.count >= self.shard_tokens:
            self._flush(full=True)

    def close(self) -> int:
        if self.count:
            self._flush(full=False)
        return self.shard_index

    def _flush(self, *, full: bool) -> None:
        tokens = np.concatenate(self.parts) if self.parts else np.zeros(0, dtype=np.uint16)
        if full and len(tokens) > self.shard_tokens:
            head = tokens[: self.shard_tokens]
            tail = tokens[self.shard_tokens :]
            self._write(head)
            self.parts = [tail]
            self.count = len(tail)
            return
        if len(tokens):
            self._write(tokens)
        self.parts = []
        self.count = 0

    def _write(self, tokens: np.ndarray) -> None:
        path = self.data_dir / f"corpus_{self.split}_{self.shard_index:04d}.npy"
        tmp = path.with_name(path.name + ".partial")
        with tmp.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(tokens, dtype=np.uint16))
        tmp.replace(path)
        print(
            f"wrote {path.name} ({len(tokens):,} tokens)",
            flush=True,
        )
        self.shard_index += 1


def _iter_docs(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)["text"]


def _take_bytes(iterator, nbytes: int) -> tuple[list[str], int]:
    docs: list[str] = []
    got = 0
    for text in iterator:
        docs.append(text)
        got += len(text)
        if got >= nbytes:
            break
    return docs, got


def encode_corpus(cfg: PrepareConfig) -> dict:
    from mork.tokenizer import MorkTokenizer

    manifest_path = cfg.data_dir / "manifest.json"
    if manifest_path.exists() and not cfg.force_encode:
        manifest = _read_json(manifest_path)
        if manifest.get("complete"):
            print(f"shards ready: {manifest_path}", flush=True)
            return manifest

    for path in cfg.data_dir.glob("corpus_*.npy"):
        path.unlink()
    for path in cfg.data_dir.glob("corpus_*.npy.partial"):
        path.unlink()

    tokenizer = MorkTokenizer.load(cfg.data_dir / "tokenizer.json")
    end_id = tokenizer.special_token_id("<|end|>")
    if end_id > np.iinfo(np.uint16).max:
        raise RuntimeError(f"<|end|> id {end_id} does not fit in uint16")

    mix = (
        ("python", 3),
        ("cosmopedia", 3),
        ("fineweb", 2),
    )
    writers = {
        split: ShardWriter(cfg.data_dir, split, cfg.shard_tokens)
        for split in ("train", "val")
    }
    for split in ("train", "val"):
        iterators = [
            _iter_docs(cfg.raw_dir / f"{name}_{split}.jsonl") for name, _ in mix
        ]
        pending = [True, True, True]
        while any(pending):
            batch: list[str] = []
            for index, (_, weight) in enumerate(mix):
                if not pending[index]:
                    continue
                docs, got = _take_bytes(iterators[index], ROUND_UNIT_BYTES * weight)
                if got == 0:
                    pending[index] = False
                    continue
                batch.extend(docs)
            if not batch:
                break
            rng = np.random.default_rng(cfg.seed + writers[split].total)
            order = rng.permutation(len(batch))
            ordered = [batch[int(i)] for i in order]
            for start in range(0, len(ordered), ENCODE_BATCH):
                chunk = ordered[start : start + ENCODE_BATCH]
                for ids in tokenizer.encode_batch(chunk):
                    ids.append(end_id)
                    writers[split].add(ids)

    counts = {split: writers[split].total for split in ("train", "val")}
    shards = {split: writers[split].close() for split in ("train", "val")}
    source_stats = {}
    for name, _weight in mix:
        source_stats[name] = {}
        for split in ("train", "val"):
            meta = _read_json(cfg.raw_dir / f"{name}_{split}.meta.json")
            source_stats[name][split] = {
                "text_bytes": meta.get("text_bytes", 0),
                "docs": meta.get("docs", 0),
            }
    manifest = {
        "complete": True,
        "dataset": DATASET,
        "vocab_size": tokenizer.vocab_size,
        "sources": source_stats,
        "tokens": counts,
        "shards": shards,
    }
    _write_json(manifest_path, manifest)
    _print_sample(cfg, tokenizer, end_id)
    return manifest


def _print_sample(cfg: PrepareConfig, tokenizer, end_id: int) -> None:
    shard = cfg.data_dir / "corpus_train_0000.npy"
    if not shard.exists():
        return
    tokens = np.load(shard)[:400]
    text = tokenizer.decode(tokens.tolist())
    print("--- sample from corpus_train_0000.npy ---", flush=True)
    print(text[:1200], flush=True)
    print(f"(<|end|> id {end_id})", flush=True)


def prepare(cfg: PrepareConfig | None = None) -> None:
    cfg = cfg or PrepareConfig()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    download_python(cfg)
    stream_text_source(
        "cosmopedia",
        "cosmopedia-v2",
        cfg.cosmopedia_bytes,
        cfg.val_ratio,
        cfg.raw_dir,
    )
    stream_text_source(
        "fineweb",
        "fineweb-edu-dedup",
        cfg.fineweb_bytes,
        cfg.val_ratio,
        cfg.raw_dir,
    )
    build_tokenizer(cfg)
    manifest = encode_corpus(cfg)
    print(
        f"corpus ready: train {manifest['tokens']['train']:,} tokens, "
        f"val {manifest['tokens']['val']:,} tokens",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and tokenize the Mork-2 corpus.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--force-encode",
        action="store_true",
        help="Retrain the tokenizer and rewrite shards. Keeps downloaded text.",
    )
    parser.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    args = parser.parse_args(argv)
    try:
        prepare(
            PrepareConfig(
                data_dir=args.data_dir,
                workers=args.workers,
                force_encode=args.force_encode,
            )
        )
    except KeyboardInterrupt:
        print("\ninterrupted; re-run mork-prepare to resume", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

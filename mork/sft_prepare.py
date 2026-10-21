"""Build the SFT corpus: English instructions with Python answers, plus short chats.

Sources, all streamed from the Hub:

- OpenCoder ``educational_instruct`` (MIT). Short "write a function" tasks whose
  answers were checked against asserts. The answer kept here is the function
  itself, not the "Here is the code" wrapper.
- ``bigcode/self-oss-instruct-sc2-exec-filter-50k`` (ODC-BY). Execution-checked
  answers. The full reply is kept when it fits in the context; otherwise just
  the fenced Python.
- ``HuggingFaceTB/smol-smoltalk``. Short English conversations, the mix used for
  SmolLM2 models around this size. Code instructions in that set are skipped
  (the two sets above cover code) and so are format-constraint drills.

A row is dropped when:

- the formatted dialogue exceeds ``seq_len`` tokens (512, the training context),
- the Python imports a module outside a small standard-library list (the eval
  sandbox cannot import it either),
- it overlaps the eval sets: a defined function name from smoke, mid or any
  MBPP split; a 13-word sequence that appears in exactly one eval problem; the
  whole text of an 8-to-12 word eval problem; or a whitespace-stripped copy of
  an MBPP solution or a mid reference,
- or the same instruction was already kept.

Outputs ``data/sft/{source}_{train,val}.jsonl`` and ``manifest.json``.
One percent is not enough to read a code eval, so validation is 2 percent,
chosen by a hash of the user text. Re-running resumes finished sources.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from mork.config import PROJECT_ROOT, ModelConfig
from mork.eval_mid import MID_TASKS
from mork.tokenizer import MorkTokenizer

EDUCATIONAL_DATASET = "OpenCoder-LLM/opc-sft-stage2"
EDUCATIONAL_CONFIG = "educational_instruct"
SC2_DATASET = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
SMOLTALK_DATASET = "HuggingFaceTB/smol-smoltalk"
MBPP_DATASET = "google-research-datasets/mbpp"

SOURCES = ("educational", "sc2", "smoltalk")
SHINGLE_N = 13
SHORT_PHRASE_MIN = 8
CODE_KEY_MIN = 40
VAL_RATIO = 0.02
SEED = 1337
SMOLTALK_LIMIT = 15_000

# Modules the eval sandbox can import. Anything else is dropped from code answers.
STDLIB_OK = frozenset(
    {
        "__future__",
        "abc",
        "bisect",
        "cmath",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "textwrap",
        "typing",
        "unicodedata",
    }
)

DEF_NAME_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", re.MULTILINE)
FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+")

DROP_FIELDS = (
    "empty",
    "import",
    "name",
    "substring",
    "shingle",
    "code",
    "length",
    "dup",
    "source",
    "shape",
)


@dataclass
class FilterStats:
    seen: int = 0
    kept_train: int = 0
    kept_val: int = 0
    tokens_train: int = 0
    tokens_val: int = 0
    drops: dict[str, int] = field(default_factory=lambda: {name: 0 for name in DROP_FIELDS})

    def drop(self, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1

    def as_dict(self) -> dict:
        return {
            "seen": self.seen,
            "kept_train": self.kept_train,
            "kept_val": self.kept_val,
            "tokens_train": self.tokens_train,
            "tokens_val": self.tokens_val,
            "drops": self.drops,
        }


@dataclass
class SftConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    sources: tuple[str, ...] = SOURCES
    seq_len: int = ModelConfig.block_size
    val_ratio: float = VAL_RATIO
    seed: int = SEED
    smoltalk_limit: int = SMOLTALK_LIMIT
    # Stop each source after this many raw rows. 0 scans the whole source.
    limit: int = 0
    force: bool = False

    @property
    def sft_dir(self) -> Path:
        return self.data_dir / "sft"

    @property
    def tokenizer_path(self) -> Path:
        return self.data_dir / "tokenizer.json"


def format_dialog(user: str, assistant: str) -> str:
    """The single dialogue layout the trainer and this filter both use."""
    return f"<|user|>\n{user.strip()}\n<|assistant|>\n{assistant.strip()}<|end|>"


def normalize_text(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def shingles(text: str, n: int = SHINGLE_N) -> set[str]:
    words = normalize_text(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def code_key(text: str) -> str:
    """Identity of a code block, ignoring comments and whitespace."""
    fenced = code_blocks(text)
    body = fenced or text
    no_comments = re.sub(r"(?m)#.*$", "", body)
    return re.sub(r"\s+", "", no_comments)


def code_blocks(text: str) -> str:
    blocks = [block.strip() for block in FENCE_RE.findall(text) if block.strip()]
    return "\n\n".join(blocks)


def defined_names(text: str, entry_point: str | None = None) -> set[str]:
    names = set(DEF_NAME_RE.findall(text))
    if entry_point:
        names.add(entry_point.strip())
    return {name for name in names if name}


def third_party_import(text: str) -> str | None:
    for module in IMPORT_RE.findall(text):
        root = module.split(".", 1)[0]
        if root not in STDLIB_OK:
            return root
    return None


def _to_val(key: str, val_ratio: float) -> bool:
    return (zlib.crc32(key.encode("utf-8")) % 10_000) < int(val_ratio * 10_000)


def _user_key(user: str) -> str:
    return hashlib.sha1(normalize_text(user).encode("utf-8")).hexdigest()


class ContaminationIndex:
    """Eval problems the SFT data must not contain.

    13-word sequences that show up in more than one problem are boilerplate
    ("write a python function that takes a"), so they are not indexed.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.codes: set[str] = set()
        self.phrases: list[str] = []
        self.shingles: set[str] = set()

    def add_problem(
        self,
        name: str | None,
        text: str,
        code: str | None = None,
    ) -> None:
        if name:
            self.names.add(name)
        phrase = normalize_text(text)
        words = phrase.split()
        if SHORT_PHRASE_MIN <= len(words) < SHINGLE_N:
            self.phrases.append(phrase)
        key = code_key(code or "")
        if len(key) >= CODE_KEY_MIN:
            self.codes.add(key)

    def finish(self, problem_texts: list[str]) -> None:
        counts: Counter[str] = Counter()
        for text in problem_texts:
            counts.update(shingles(text))
        self.shingles = {gram for gram, count in counts.items() if count == 1}

    def reason(self, user: str, assistant: str, names: set[str]) -> str | None:
        if names & self.names:
            return "name"
        user_norm = normalize_text(user)
        for phrase in self.phrases:
            if phrase in user_norm:
                return "substring"
        if shingles(user) & self.shingles:
            return "shingle"
        key = code_key(assistant)
        if len(key) >= CODE_KEY_MIN and key in self.codes:
            return "code"
        return None


def _smoke_records() -> list[tuple[str, str, str]]:
    """(name, description, tests) parsed from source, so this file does not import torch."""
    path = Path(__file__).with_name("evaluate.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SMOKE_TASKS" for target in node.targets
        ):
            tasks = ast.literal_eval(node.value)
            break
    else:
        raise RuntimeError(f"SMOKE_TASKS not found in {path}")
    records = []
    for name, prompt, tests in tasks:
        description = prompt.split('"""', 2)[1].split("\n\n", 1)[0].strip()
        records.append((name, description, tests))
    return records


def _mbpp_rows() -> list[dict]:
    from datasets import load_dataset

    try:
        dataset = load_dataset(MBPP_DATASET, "full")
    except Exception:
        dataset = load_dataset(MBPP_DATASET)
    rows = []
    for split in dataset:
        rows.extend(dataset[split])
    if len(rows) < 900:
        raise RuntimeError(f"expected about 974 MBPP problems, loaded {len(rows)}")
    return rows


def _mbpp_name(row: dict) -> str | None:
    tests = " ".join(row.get("test_list") or [])
    names = DEF_NAME_RE.findall(row.get("code") or "")
    for name in names:
        if re.search(rf"\b{name}\s*\(", tests):
            return name
    return names[0] if names else None


def build_index() -> ContaminationIndex:
    index = ContaminationIndex()
    texts: list[str] = []
    for name, description, _tests in _smoke_records():
        index.add_problem(name, description)
        texts.append(description)
    for name, _params, doc, _call, _result, _asserts, reference in MID_TASKS:
        index.add_problem(name, doc, f"def {name}():\n{reference}")
        texts.append(doc)
    print("loading MBPP for decontamination...", flush=True)
    for row in _mbpp_rows():
        text = row.get("text") or row.get("prompt") or ""
        index.add_problem(_mbpp_name(row), text, row.get("code") or "")
        texts.append(text)
    index.finish(texts)
    print(
        f"eval index: {len(index.names)} names, {len(index.shingles)} shingles, "
        f"{len(index.phrases)} short phrases, {len(index.codes)} solutions",
        flush=True,
    )
    return index


def _token_count(tokenizer: MorkTokenizer, text: str, seq_len: int) -> int | None:
    if len(text.encode("utf-8")) > seq_len * 12:
        return None
    count = len(tokenizer.encode(text))
    if count > seq_len:
        return None
    return count


def _record(
    source: str,
    user: str,
    assistant: str,
    n_tokens: int,
    extra: dict | None = None,
) -> dict:
    row = {
        "source": source,
        "user": user,
        "assistant": assistant,
        "n_tokens": n_tokens,
    }
    if extra:
        row.update(extra)
    return row


class _Writer:
    def __init__(self, directory: Path, source: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.paths = {
            split: directory / f"{source}_{split}.jsonl" for split in ("train", "val")
        }
        self._handles = {
            split: path.with_suffix(".jsonl.partial").open("w", encoding="utf-8")
            for split, path in self.paths.items()
        }

    def add(self, split: str, row: dict) -> None:
        self._handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")

    def finish(self) -> None:
        for split, handle in self._handles.items():
            handle.close()
            partial = self.paths[split].with_suffix(".jsonl.partial")
            partial.replace(self.paths[split])


def _keep(
    stats: FilterStats,
    writer: _Writer,
    seen: set[str],
    cfg: SftConfig,
    source: str,
    user: str,
    assistant: str,
    n_tokens: int,
    extra: dict | None = None,
) -> None:
    key = _user_key(user)
    if key in seen:
        stats.drop("dup")
        return
    seen.add(key)
    split = "val" if _to_val(key, cfg.val_ratio) else "train"
    writer.add(split, _record(source, user, assistant, n_tokens, extra))
    if split == "val":
        stats.kept_val += 1
        stats.tokens_val += n_tokens
    else:
        stats.kept_train += 1
        stats.tokens_train += n_tokens


def _answer_fits(
    tokenizer: MorkTokenizer,
    seq_len: int,
    user: str,
    assistant: str,
) -> int | None:
    return _token_count(tokenizer, format_dialog(user, assistant), seq_len)


def _answer_options(full: str, code: str, *, code_first: bool) -> list[str]:
    """Unique reply candidates. Code-first keeps the bare function when it fits."""
    preferred = (code, full) if code_first else (full, code)
    ordered: list[str] = []
    for candidate in preferred:
        text = (candidate or "").strip()
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def prepare_educational(
    cfg: SftConfig,
    tokenizer: MorkTokenizer,
    index: ContaminationIndex,
    seen: set[str],
) -> FilterStats:
    from datasets import load_dataset

    stats = FilterStats()
    writer = _Writer(cfg.sft_dir, "educational")
    dataset = load_dataset(
        EDUCATIONAL_DATASET, EDUCATIONAL_CONFIG, split="train", streaming=True
    )
    t0 = time.time()
    try:
        for row in dataset:
            if cfg.limit and stats.seen >= cfg.limit:
                break
            stats.seen += 1
            user = (row.get("instruction") or "").strip()
            code = (row.get("code") or "").strip()
            full = (row.get("output") or "").strip()
            if not user or not (code or full):
                stats.drop("empty")
                continue
            names = defined_names(code or full, str(row.get("entry_point") or ""))
            reason = index.reason(user, code or full, names)
            if reason:
                stats.drop(reason)
                continue
            answer = _choose_answer(
                tokenizer, cfg.seq_len, user, _answer_options(full, code, code_first=True)
            )
            if isinstance(answer, str):
                stats.drop(answer)
                continue
            text, n_tokens = answer
            tests = row.get("testcase") or []
            if isinstance(tests, str):
                tests = [tests]
            _keep(
                stats,
                writer,
                seen,
                cfg,
                "educational",
                user,
                text,
                n_tokens,
                {
                    "entry_point": str(row.get("entry_point") or ""),
                    "tests": [str(item) for item in tests],
                },
            )
            _progress("educational", stats, t0)
    finally:
        writer.finish()
    _progress("educational", stats, t0, force=True)
    return stats


def prepare_sc2(
    cfg: SftConfig,
    tokenizer: MorkTokenizer,
    index: ContaminationIndex,
    seen: set[str],
) -> FilterStats:
    from datasets import load_dataset

    stats = FilterStats()
    writer = _Writer(cfg.sft_dir, "sc2")
    dataset = load_dataset(SC2_DATASET, split="train", streaming=True)
    t0 = time.time()
    try:
        for row in dataset:
            if cfg.limit and stats.seen >= cfg.limit:
                break
            stats.seen += 1
            user = (row.get("instruction") or "").strip()
            full = (row.get("response") or "").strip()
            if not user or not full:
                stats.drop("empty")
                continue
            fenced = code_blocks(full)
            names = defined_names(full) | defined_names(fenced)
            reason = index.reason(user, fenced or full, names)
            if reason:
                stats.drop(reason)
                continue
            # Keep the explanation when it fits; the fenced function is the fallback.
            answer = _choose_answer(
                tokenizer, cfg.seq_len, user, _answer_options(full, fenced, code_first=False)
            )
            if isinstance(answer, str):
                stats.drop(answer)
                continue
            text, n_tokens = answer
            _keep(stats, writer, seen, cfg, "sc2", user, text, n_tokens)
            _progress("sc2", stats, t0)
    finally:
        writer.finish()
    _progress("sc2", stats, t0, force=True)
    return stats


def _choose_answer(
    tokenizer: MorkTokenizer,
    seq_len: int,
    user: str,
    candidates: list[str],
) -> tuple[str, int] | str:
    saw_import = False
    for candidate in candidates:
        if third_party_import(candidate):
            saw_import = True
            continue
        count = _answer_fits(tokenizer, seq_len, user, candidate)
        if count is None:
            continue
        return candidate, count
    return "import" if saw_import else "length"


def _smoltalk_pair(messages: list) -> tuple[str, str] | None:
    if not messages or not isinstance(messages, list):
        return None
    roles = [item.get("role") for item in messages]
    contents = [(item.get("content") or "").strip() for item in messages]
    if roles == ["user", "assistant"] and all(contents):
        return contents[0], contents[1]
    if roles == ["system", "user", "assistant"] and all(contents):
        return f"{contents[0]}\n\n{contents[1]}", contents[2]
    return None


def _smoltalk_skipped(source: str) -> bool:
    name = source.lower()
    return "self-oss" in name or "constraint" in name or "contraint" in name


def _take_share(groups: dict[str, list], limit: int) -> list[dict]:
    """Up to ``limit`` rows, spread evenly. ``limit <= 0`` keeps every row."""
    if limit <= 0:
        rows: list[dict] = []
        for name in sorted(groups):
            rows.extend(groups[name])
        return rows
    remaining = {name: list(rows) for name, rows in groups.items()}
    chosen: dict[str, list] = {name: [] for name in groups}
    left = limit
    while left > 0 and any(remaining.values()):
        active = sorted(
            (name for name, rows in remaining.items() if rows),
            key=lambda name: len(remaining[name]),
            reverse=True,
        )
        share = max(1, left // len(active))
        for name in active:
            if left <= 0:
                break
            take = min(share, len(remaining[name]), left)
            chosen[name].extend(remaining[name][:take])
            remaining[name] = remaining[name][take:]
            left -= take
    rows = []
    for name in sorted(chosen):
        rows.extend(chosen[name])
    return rows


def prepare_smoltalk(
    cfg: SftConfig,
    tokenizer: MorkTokenizer,
    index: ContaminationIndex,
    seen: set[str],
) -> FilterStats:
    from datasets import load_dataset

    stats = FilterStats()
    dataset = load_dataset(SMOLTALK_DATASET, split="train", streaming=True)
    rng = random.Random(cfg.seed)
    # One reservoir per subset. Capped so one chatty subset cannot fill memory;
    # a limit of 0 keeps every row that passed the filters.
    cap = cfg.smoltalk_limit if cfg.smoltalk_limit > 0 else 0
    buckets: dict[str, list[dict]] = {}
    bucket_seen: Counter[str] = Counter()
    held: set[str] = set()
    t0 = time.time()
    for row in dataset:
        if cfg.limit and stats.seen >= cfg.limit:
            break
        stats.seen += 1
        source_name = str(row.get("source") or "unknown")
        if _smoltalk_skipped(source_name):
            stats.drop("source")
            continue
        pair = _smoltalk_pair(row.get("messages"))
        if pair is None:
            stats.drop("shape")
            continue
        user, assistant = pair
        reason = index.reason(user, assistant, defined_names(assistant))
        if reason:
            stats.drop(reason)
            continue
        count = _answer_fits(tokenizer, cfg.seq_len, user, assistant)
        if count is None:
            stats.drop("length")
            continue
        key = _user_key(user)
        if key in seen or key in held:
            stats.drop("dup")
            continue
        item = _record("smoltalk", user, assistant, count, {"subset": source_name, "_key": key})
        bucket = buckets.setdefault(source_name, [])
        bucket_seen[source_name] += 1
        seen_n = bucket_seen[source_name]
        if cap <= 0 or len(bucket) < cap:
            bucket.append(item)
            held.add(key)
        else:
            slot = rng.randrange(seen_n)
            if slot < cap:
                held.discard(bucket[slot]["_key"])
                bucket[slot] = item
                held.add(key)
        if stats.seen % 20_000 == 0:
            print(
                f"  smoltalk scanned {stats.seen:,} "
                f"({sum(len(v) for v in buckets.values()):,} in reservoirs, "
                f"{time.time() - t0:.0f}s)",
                flush=True,
            )

    picked = _take_share(buckets, cfg.smoltalk_limit)
    writer = _Writer(cfg.sft_dir, "smoltalk")
    try:
        for item in picked:
            key = item.pop("_key")
            seen.add(key)
            split = "val" if _to_val(key, cfg.val_ratio) else "train"
            writer.add(split, item)
            if split == "val":
                stats.kept_val += 1
                stats.tokens_val += item["n_tokens"]
            else:
                stats.kept_train += 1
                stats.tokens_train += item["n_tokens"]
    finally:
        writer.finish()
    _progress("smoltalk", stats, t0, force=True)
    print(
        "  smoltalk subsets: "
        + ", ".join(f"{name} {len(rows):,}" for name, rows in sorted(buckets.items())),
        flush=True,
    )
    return stats


def _progress(source: str, stats: FilterStats, t0: float, force: bool = False) -> None:
    if not force and stats.seen % 5_000 != 0:
        return
    kept = stats.kept_train + stats.kept_val
    print(
        f"  {source} seen {stats.seen:,} kept {kept:,} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )


def _source_done(cfg: SftConfig, source: str) -> bool:
    manifest = _read_manifest(cfg)
    entry = manifest.get("sources", {}).get(source)
    if not entry or not entry.get("complete"):
        return False
    return all((cfg.sft_dir / f"{source}_{split}.jsonl").exists() for split in ("train", "val"))


def _read_manifest(cfg: SftConfig) -> dict:
    path = cfg.sft_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(cfg: SftConfig, payload: dict) -> None:
    path = cfg.sft_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def prepare_sft(cfg: SftConfig | None = None) -> dict:
    cfg = cfg or SftConfig()
    if not cfg.tokenizer_path.exists():
        raise SystemExit(f"tokenizer not found: {cfg.tokenizer_path}")
    unknown = [name for name in cfg.sources if name not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source {unknown[0]!r} (choose from {', '.join(SOURCES)})")

    cfg.sft_dir.mkdir(parents=True, exist_ok=True)
    pending = [name for name in cfg.sources if cfg.force or not _source_done(cfg, name)]
    if not pending:
        print(f"sft data ready: {cfg.sft_dir / 'manifest.json'}", flush=True)
        return _read_manifest(cfg)

    tokenizer = MorkTokenizer.load(cfg.tokenizer_path)
    index = build_index()
    seen: set[str] = set()
    # Dedup against sources that were already written and are not being rebuilt.
    for name in SOURCES:
        if name in pending:
            continue
        for split in ("train", "val"):
            path = cfg.sft_dir / f"{name}_{split}.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        seen.add(_user_key(json.loads(line)["user"]))

    runners = {
        "educational": prepare_educational,
        "sc2": prepare_sc2,
        "smoltalk": prepare_smoltalk,
    }
    manifest = _read_manifest(cfg)
    manifest["sources"] = manifest.get("sources", {})
    manifest.update(
        {
            "seq_len": cfg.seq_len,
            "val_ratio": cfg.val_ratio,
            "seed": cfg.seed,
            "smoltalk_limit": cfg.smoltalk_limit,
            "decontamination": {
                "names": len(index.names),
                "shingles": len(index.shingles),
                "short_phrases": len(index.phrases),
                "solutions": len(index.codes),
            },
        }
    )
    for name in pending:
        print(f"preparing {name}...", flush=True)
        stats = runners[name](cfg, tokenizer, index, seen)
        manifest["sources"][name] = {"complete": cfg.limit == 0, **stats.as_dict()}
        _write_manifest(cfg, manifest)
    kept = sum(
        entry.get("kept_train", 0) + entry.get("kept_val", 0)
        for entry in manifest["sources"].values()
    )
    print(f"sft corpus: {kept:,} examples in {cfg.sft_dir}", flush=True)
    return manifest


def _self_check() -> None:
    index = ContaminationIndex()
    texts = [
        "Write a function to check whether a given string is a palindrome ignoring case and spaces completely.",
        "Write a function to check whether a given string is a palindrome ignoring case but treat digits as letters.",
    ]
    solution = (
        "def is_palindrome(text):\n"
        "    cleaned = ''.join(ch for ch in text.lower() if ch.isalnum())\n"
        "    return cleaned == cleaned[::-1]\n"
    )
    index.add_problem("is_palindrome", texts[0], solution)
    index.add_problem("is_near_palindrome", texts[1])
    index.add_problem("add", "Return the sum of a and b.")
    index.finish(texts + ["Return the sum of a and b."])

    shared = shingles(texts[0]) & shingles(texts[1])
    if not shared:
        raise SystemExit("fixture should share a boilerplate 13-word sequence")
    if shared & index.shingles:
        raise SystemExit(f"boilerplate shingle was indexed: {shared & index.shingles}")
    copied_prompt = texts[0] + " Implement it in Python."
    if index.reason(copied_prompt, "def is_palindrome(text):\n    return True\n", {"is_palindrome"}) != "name":
        raise SystemExit("name overlap was not caught")
    if index.reason(copied_prompt, "def check(text):\n    return True\n", {"check"}) != "shingle":
        raise SystemExit("13-word overlap was not caught")
    if index.reason("Return the sum of a and b.", "def add(a, b):\n    return a + b\n", set()) is not None:
        raise SystemExit("short prompt without a copied name should pass")
    if index.reason("Return the sum of a and b.", "def add(a, b):\n    return a + b\n", {"add"}) != "name":
        raise SystemExit("eval function name should be dropped")
    if index.reason("Unrelated request about cleaning strings.", solution, {"check"}) != "code":
        raise SystemExit("copied solution was not caught")
    if third_party_import("import numpy as np\nimport math\n") != "numpy":
        raise SystemExit("numpy should be rejected")
    if third_party_import("import math\nfrom collections import Counter\n") is not None:
        raise SystemExit("stdlib imports should pass")
    pair = _smoltalk_pair(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    )
    if pair != ("Be brief.\n\nHi", "Hello"):
        raise SystemExit(f"system message not folded: {pair}")
    if _smoltalk_pair([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "More"}]):
        raise SystemExit("multi-turn should be dropped")
    groups = {"a": [{"n": i} for i in range(1)], "b": [{"n": i} for i in range(10)], "c": [{"n": i} for i in range(10)]}
    picked = _take_share(groups, 7)
    if len(picked) != 7:
        raise SystemExit(f"share kept {len(picked)}")
    if _smoltalk_skipped("smol-contraints") is not True or _smoltalk_skipped("smol-magpie-ultra-short"):
        raise SystemExit("source skip rules are wrong")
    smoke = _smoke_records()
    if len(smoke) != 20 or smoke[0][0] != "add" or "sum" not in smoke[0][1].lower():
        raise SystemExit(f"smoke parse failed: {smoke[:1]}")
    tokenizer_path = PROJECT_ROOT / "data" / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = MorkTokenizer.load(tokenizer_path)
        text = format_dialog("Say hi", "Hello")
        ids = tokenizer.encode(text)
        if tokenizer.special_token_id("<|user|>") not in ids or tokenizer.special_token_id("<|end|>") not in ids:
            raise SystemExit("special tokens were split apart")
        if tokenizer.decode(ids) != text:
            raise SystemExit(f"dialogue did not round-trip: {tokenizer.decode(ids)!r}")
    print("sft filter self-check passed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and filter the Mork-2 SFT data.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--seq-len", type=int, default=ModelConfig.block_size)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument(
        "--smoltalk-limit",
        type=int,
        default=SMOLTALK_LIMIT,
        help="Max kept smoltalk chats, spread across subsets (0 keeps all that fit).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max raw rows to scan per source.")
    parser.add_argument("--force", action="store_true", help="Rebuild sources that already finished.")
    parser.add_argument("--self-check", action="store_true", help="Run the filter checks and exit.")
    args = parser.parse_args(argv)
    if args.self_check:
        _self_check()
        return
    try:
        prepare_sft(
            SftConfig(
                data_dir=args.data_dir,
                sources=tuple(args.sources),
                seq_len=args.seq_len,
                val_ratio=args.val_ratio,
                smoltalk_limit=args.smoltalk_limit,
                limit=args.limit,
                force=args.force,
            )
        )
    except KeyboardInterrupt:
        print("\ninterrupted; re-run mork-sft-prepare to resume", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

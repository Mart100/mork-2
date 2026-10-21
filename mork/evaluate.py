"""Milestone evaluation for Mork-2 checkpoints: held-out loss and Python functions.

Code tasks default to completion: the model continues a function header and
docstring. The body is cut at the first unindented line, then run with the
task's asserts in a subprocess (``python -s -P``, empty env except a fixed
hash seed, temp cwd, 5 s timeout). This is isolation from the eval process,
not a security sandbox.

``--format instruct`` asks for the whole function after ``<|assistant|>``.
Those rows are stored as ``smoke-instruct``, ``mid-instruct``, and
``mbpp-instruct``. The completion rows are unchanged.

Task sets:
  loss   Loss on the first --val-tokens of the shuffled val stream (a superset
         of the slice the trainer scores), plus per-source loss on windows of
         data/raw/{python,cosmopedia,fineweb}_val.jsonl.
  smoke  20 hand-written beginner functions, 10 samples each by default.
  mid    160 hand-written mid-difficulty functions (mork/eval_mid.py),
         5 samples each by default.
  mbpp   MBPP sanitized test split (257 tasks), 5 samples each by default.
         Evaluation only; never put MBPP into training data.

Each checkpoint appends one row per task set to ``loss_eval.csv`` or
``code_eval.csv`` in the run that wrote the checkpoint, or in ``--out-dir``.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from mork.checkpoint import load_model_for_inference
from mork.config import PROJECT_ROOT, TrainConfig
from mork.data import DataLoaderLite
from mork.eval_mid import mid_task_items
from mork.generate import DEFAULT_TOKENIZER_PATH, toplevel_cut
from mork.model import GPT
from mork.tokenizer import MorkTokenizer

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "latest.pt"
FALLBACK_DIR = PROJECT_ROOT / "runs"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_SAMPLES = {"smoke": 10, "mid": 5, "mbpp": 5}
DEFAULT_MAX_NEW_TOKENS = {"smoke": 128, "mid": 160, "mbpp": 160}

SOURCES = ("python", "cosmopedia", "fineweb")
LOSS_BATCH = 8
SOURCE_WINDOWS = 200
SOURCE_DOCS = 160
SOURCE_MAX_DOC_TOKENS = 1024

LOSS_CSV_FIELDS = [
    "time",
    "checkpoint",
    "step",
    "val_tokens",
    "val_loss",
    "val_train_slice",
    *SOURCES,
    "elapsed_s",
]

CSV_FIELDS = [
    "time",
    "checkpoint",
    "step",
    "tasks",
    "n_tasks",
    "samples",
    "temperature",
    "top_k",
    "greedy_pass",
    "greedy_syntax",
    "sample_pass_at_1",
    "sample_syntax_rate",
    "tasks_solved",
    "elapsed_s",
]


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str  # ends right after the closing docstring quotes, no newline
    tests: str
    setup: str = ""  # imports needed by the tests, placed above the prompt


SMOKE_TASKS = [
    ("add", 'def add(a, b):\n    """Return the sum of a and b.\n\n    >>> add(2, 3)\n    5\n    """',
     "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(10, -4) == 6\n"),
    ("is_even", 'def is_even(n):\n    """Return True if n is even, otherwise False.\n\n    >>> is_even(4)\n    True\n    """',
     "assert is_even(4) == True\nassert is_even(7) == False\nassert is_even(0) == True\n"),
    ("find_max", 'def find_max(numbers):\n    """Return the largest number in the list.\n\n    >>> find_max([3, 7, 2])\n    7\n    """',
     "assert find_max([3, 7, 2]) == 7\nassert find_max([-5, -2, -9]) == -2\nassert find_max([4]) == 4\n"),
    ("reverse_string", 'def reverse_string(s):\n    """Return the string s reversed.\n\n    >>> reverse_string("hello")\n    \'olleh\'\n    """',
     "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nassert reverse_string('ab') == 'ba'\n"),
    ("factorial", 'def factorial(n):\n    """Return n factorial.\n\n    >>> factorial(5)\n    120\n    """',
     "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(3) == 6\n"),
    ("fibonacci", 'def fibonacci(n):\n    """Return the nth Fibonacci number, with fibonacci(0) = 0 and fibonacci(1) = 1.\n\n    >>> fibonacci(10)\n    55\n    """',
     "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55\n"),
    ("count_vowels", 'def count_vowels(s):\n    """Return the number of vowels (a, e, i, o, u) in the lowercase string s.\n\n    >>> count_vowels("hello")\n    2\n    """',
     "assert count_vowels('hello') == 2\nassert count_vowels('sky') == 0\nassert count_vowels('banana') == 3\n"),
    ("is_palindrome", 'def is_palindrome(s):\n    """Return True if s reads the same forwards and backwards.\n\n    >>> is_palindrome("racecar")\n    True\n    """',
     "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('a') == True\n"),
    ("sum_list", 'def sum_list(numbers):\n    """Return the sum of all numbers in the list.\n\n    >>> sum_list([1, 2, 3])\n    6\n    """',
     "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0\nassert sum_list([-2, 2, 5]) == 5\n"),
    ("celsius_to_fahrenheit", 'def celsius_to_fahrenheit(c):\n    """Convert a temperature from Celsius to Fahrenheit.\n\n    >>> celsius_to_fahrenheit(100)\n    212.0\n    """',
     "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212\n"),
    ("square_all", 'def square_all(numbers):\n    """Return a new list with each number squared.\n\n    >>> square_all([1, 2, 3])\n    [1, 4, 9]\n    """',
     "assert square_all([1, 2, 3]) == [1, 4, 9]\nassert square_all([]) == []\nassert square_all([-2]) == [4]\n"),
    ("remove_duplicates", 'def remove_duplicates(items):\n    """Return a list with duplicates removed, keeping the original order.\n\n    >>> remove_duplicates([1, 2, 2, 3, 1])\n    [1, 2, 3]\n    """',
     "assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]\nassert remove_duplicates([]) == []\nassert remove_duplicates(['b', 'a', 'b']) == ['b', 'a']\n"),
    ("is_prime", 'def is_prime(n):\n    """Return True if n is a prime number.\n\n    >>> is_prime(7)\n    True\n    """',
     "assert is_prime(2) == True\nassert is_prime(9) == False\nassert is_prime(13) == True\nassert is_prime(1) == False\n"),
    ("find_min", 'def find_min(numbers):\n    """Return the smallest number in the list.\n\n    >>> find_min([4, 1, 8])\n    1\n    """',
     "assert find_min([4, 1, 8]) == 1\nassert find_min([-3, 5]) == -3\nassert find_min([9]) == 9\n"),
    ("count_words", 'def count_words(sentence):\n    """Return the number of words in the sentence.\n\n    >>> count_words("the quick brown fox")\n    4\n    """',
     "assert count_words('the quick brown fox') == 4\nassert count_words('hello') == 1\n"),
    ("average", 'def average(numbers):\n    """Return the average of a non-empty list of numbers.\n\n    >>> average([1, 2, 3, 4])\n    2.5\n    """',
     "assert average([1, 2, 3, 4]) == 2.5\nassert average([5]) == 5\n"),
    ("get_evens", 'def get_evens(numbers):\n    """Return a list of the even numbers in numbers.\n\n    >>> get_evens([1, 2, 3, 4, 6])\n    [2, 4, 6]\n    """',
     "assert get_evens([1, 2, 3, 4, 6]) == [2, 4, 6]\nassert get_evens([1, 3]) == []\n"),
    ("multiply", 'def multiply(a, b):\n    """Return the product of a and b.\n\n    >>> multiply(3, 4)\n    12\n    """',
     "assert multiply(3, 4) == 12\nassert multiply(-2, 5) == -10\nassert multiply(0, 7) == 0\n"),
    ("max_of_three", 'def max_of_three(a, b, c):\n    """Return the largest of the three numbers a, b and c.\n\n    >>> max_of_three(1, 5, 3)\n    5\n    """',
     "assert max_of_three(1, 5, 3) == 5\nassert max_of_three(9, 2, 4) == 9\nassert max_of_three(1, 2, 8) == 8\n"),
    ("count_occurrences", 'def count_occurrences(items, target):\n    """Return how many times target appears in items.\n\n    >>> count_occurrences([1, 2, 1, 1], 1)\n    3\n    """',
     "assert count_occurrences([1, 2, 1, 1], 1) == 3\nassert count_occurrences([], 5) == 0\nassert count_occurrences(['a', 'b'], 'b') == 1\n"),
]

DEF_RE = re.compile(r"^def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*[^:]+)?:", re.MULTILINE)
IMPORT_RE = re.compile(r"^(?:import|from)\s+\S.*$", re.MULTILINE)


def smoke_tasks() -> list[Task]:
    return [Task(name, prompt, tests) for name, prompt, tests in SMOKE_TASKS]


def mid_tasks() -> list[Task]:
    return [Task(name, prompt, tests) for name, prompt, tests in mid_task_items()]


def mbpp_prompt(row: dict) -> str | None:
    """Reference imports, the tested function's def line, problem text and first assert."""
    first_test = row["test_list"][0]
    defs = DEF_RE.findall(row["code"])
    target = next(
        ((name, params) for name, params in defs if re.search(rf"\b{name}\s*\(", first_test)),
        None,
    )
    if target is None:
        return None
    name, params = target
    imports = IMPORT_RE.findall(row["code"])
    header = "".join(line.strip() + "\n" for line in imports)
    if header:
        header += "\n"
    text = " ".join(row["prompt"].split())
    return (
        f"{header}def {name}({params}):\n"
        f'    """{text}\n\n'
        f"    {first_test.strip()}\n"
        f'    """'
    )


def mbpp_tasks() -> list[Task]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    tasks = []
    skipped = 0
    for row in ds:
        prompt = mbpp_prompt(row)
        if prompt is None:
            skipped += 1
            continue
        tasks.append(
            Task(
                name=f"mbpp_{row['task_id']}",
                prompt=prompt,
                tests="\n".join(row["test_list"]) + "\n",
                setup="".join(line + "\n" for line in row["test_imports"]),
            )
        )
    if skipped:
        print(f"mbpp: skipped {skipped} tasks without a matching def line")
    return tasks


def extract_body(completion: str) -> str:
    text = completion.split("<|end|>")[0]
    cut = toplevel_cut(text)
    if cut is not None:
        text = text[:cut]
    return text.rstrip() + "\n"


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_START_RE = re.compile(r"(?m)^(import |from |def |async def |class |@)")
_DOC_RE = re.compile(r'"""(.*?)"""', re.DOTALL)
_DOCTEST_RE = re.compile(r">>>\s*(.+)\n[ \t]*(.+)")
_ASSERT_RE = re.compile(r"assert\s+(.+?)\s*==\s*(.+)")


def instruction_from_stub(prompt: str) -> str:
    """Prose instruction for a completion stub, in the SFT user-turn style."""
    match = DEF_RE.search(prompt)
    if match is None:
        raise ValueError("stub has no def line")
    name, params = match.group(1), match.group(2).strip()
    doc_match = _DOC_RE.search(prompt)
    doc = doc_match.group(1) if doc_match else ""
    prose, call, result = _split_doc(doc)
    sentence = f"Write a Python function `{name}({params})`."
    if prose:
        sentence += f" {prose}"
    if call is not None and result is not None:
        sentence += f" Example: `{call}` returns `{result}`."
    return sentence


def _split_doc(doc: str) -> tuple[str, str | None, str | None]:
    doctest = _DOCTEST_RE.search(doc)
    if doctest:
        prose = " ".join(doc[: doctest.start()].split())
        return prose, doctest.group(1).strip(), doctest.group(2).strip()
    assertion = _ASSERT_RE.search(doc)
    if assertion:
        prose = " ".join(doc[: assertion.start()].split())
        return prose, assertion.group(1).strip(), assertion.group(2).strip()
    return " ".join(doc.split()), None, None


def instruct_prompt(instruction: str) -> str:
    return f"<|user|>\n{instruction.strip()}\n<|assistant|>\n"


def extract_source(completion: str) -> str:
    """Whole program from an instruct completion: a fence, or the first function."""
    text = completion.split("<|end|>", 1)[0]
    fences = [block.strip() for block in _FENCE_RE.findall(text) if block.strip()]
    if fences:
        body = "\n\n".join(fences)
    else:
        match = _CODE_START_RE.search(text)
        body = text[match.start() :] if match else text
        body = _cut_after_program(body)
    return body.rstrip() + "\n"


def _cut_after_program(text: str) -> str:
    """Keep leading imports and the first function. Drop a later top-level line."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    saw_def = False
    keep = len(lines)
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        is_header = stripped.startswith(("import ", "from ", "def ", "async def ", "class ", "@"))
        if i == 0:
            if stripped.startswith(("def ", "async def ", "class ")):
                saw_def = True
            continue
        if not line.strip() or line[0].isspace():
            continue
        if not saw_def and is_header:
            if stripped.startswith(("def ", "async def ", "class ")):
                saw_def = True
            continue
        keep = i
        break
    return "".join(lines[:keep])


def run_program(program: str, workdir: str) -> tuple[bool, bool]:
    """Returns (compiles, passes)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(program, "<gen>", "exec")
    except Exception:
        return False, False
    try:
        # Like -I, but with a fixed hash seed so set/dict-order answers pass or fail consistently.
        result = subprocess.run(
            [sys.executable, "-s", "-P", "-c", program],
            cwd=workdir,
            env={"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", "")},
            capture_output=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return True, False
    return True, result.returncode == 0


@torch.no_grad()
def sample(
    model: GPT,
    tokenizer: MorkTokenizer,
    prompt_ids: list[int],
    *,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    seed: int,
    stop_at_toplevel: bool = True,
) -> list[str]:
    """``n`` completions in one batch; ``temperature == 0`` is greedy.

    Completion mode stops at the first new top-level line. Instruct mode
    stops only at ``<|end|>``, because the reply is a whole function.
    """
    end_id = tokenizer.special_token_id("<|end|>")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    idx = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    rows = list(range(n))  # original sample index of each row still in idx
    done: dict[int, list[int]] = {}
    kv_cache = model.new_kv_cache()
    n_new = prompt_len
    for step in range(max_new_tokens):
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model.next_logits(idx, kv_cache, n_new)
        n_new = 1
        logits = logits.float()
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=gen)
        idx = torch.cat((idx, nxt), dim=1)
        if step % 8 == 7:
            keep = []
            for j, out in enumerate(idx[:, prompt_len:].tolist()):
                finished = end_id in out or (
                    stop_at_toplevel and toplevel_cut(tokenizer.decode(out)) is not None
                )
                if finished:
                    done[rows[j]] = out
                else:
                    keep.append(j)
            if not keep:
                break
            if len(keep) < len(rows):
                idx = idx[keep]
                rows = [rows[j] for j in keep]
                kv_cache.keep_rows(torch.tensor(keep, device=idx.device))
    for j, out in enumerate(idx[:, prompt_len:].tolist()):
        done.setdefault(rows[j], out)
    return [tokenizer.decode(done[i]) for i in range(n)]


def evaluate_tasks(
    model: GPT,
    tokenizer: MorkTokenizer,
    tasks: list[Task],
    *,
    samples: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    device: str,
    seed: int,
    verbose: bool,
    workers: int = 8,
    format: str = "completion",
) -> tuple[dict, list[dict]]:
    per_task = []
    # Tests run in a thread pool while the GPU generates the next task.
    pending: list[tuple[Task, str, list[Future]]] = []
    instruct = format == "instruct"
    with tempfile.TemporaryDirectory() as workdir, ThreadPoolExecutor(workers) as pool:
        for i, task in enumerate(tasks):
            prompt = instruct_prompt(instruction_from_stub(task.prompt)) if instruct else task.prompt
            ids = tokenizer.encode(prompt)
            common = dict(
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                device=device,
                stop_at_toplevel=not instruct,
            )
            outs = sample(model, tokenizer, ids, n=1, temperature=0.0, seed=seed, **common)
            if samples > 0:
                outs += sample(
                    model, tokenizer, ids, n=samples, temperature=temperature,
                    seed=seed + 1000 + i, **common,
                )
            if instruct:
                bodies = [extract_source(out) for out in outs]
                programs = [task.setup + body + "\n" + task.tests for body in bodies]
            else:
                bodies = [extract_body(out) for out in outs]
                programs = [
                    task.setup + task.prompt + body + "\n\n" + task.tests for body in bodies
                ]
            futures = [pool.submit(run_program, program, workdir) for program in programs]
            pending.append((task, bodies[0], futures))
            if not verbose and (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(tasks)} tasks generated", flush=True)

        for task, greedy_body, futures in pending:
            results = [f.result() for f in futures]
            g_syntax, g_pass = results[0]
            s_syntax = sum(r[0] for r in results[1:])
            s_pass = sum(r[1] for r in results[1:])
            per_task.append(
                {
                    "task": task.name,
                    "greedy_syntax": g_syntax,
                    "greedy_pass": g_pass,
                    "sample_syntax": s_syntax,
                    "sample_pass": s_pass,
                    "greedy_body": greedy_body,
                }
            )
            if verbose:
                print(
                    f"  {task.name:<24} greedy {'pass' if g_pass else 'fail'}  "
                    f"sampled {s_pass}/{samples}",
                    flush=True,
                )

    n = len(tasks)
    n_samples = max(1, n * samples)
    summary = {
        "n_tasks": n,
        "greedy_pass": sum(t["greedy_pass"] for t in per_task),
        "greedy_syntax": sum(t["greedy_syntax"] for t in per_task),
        "sample_pass_at_1": sum(t["sample_pass"] for t in per_task) / n_samples,
        "sample_syntax_rate": sum(t["sample_syntax"] for t in per_task) / n_samples,
        "tasks_solved": sum(1 for t in per_task if t["sample_pass"] > 0),
    }
    return summary, per_task


def checkpoint_run_dir(path: Path) -> Path | None:
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    run_dir = ckpt.get("run_dir")
    if run_dir is None or not Path(run_dir).is_dir():
        return None
    return Path(run_dir)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _ckpt_key(path: Path) -> str:
    return str(path.expanduser().resolve())


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _eval_csv_paths(filename: str, out_dir: Path | None = None) -> list[Path]:
    """CSVs to consult for skip: optional out_dir first, then every runs/*/filename."""
    seen: set[Path] = set()
    paths: list[Path] = []
    candidates = []
    if out_dir is not None:
        candidates.append(out_dir / filename)
    candidates.extend(sorted(FALLBACK_DIR.glob(f"*/{filename}")))
    for path in candidates:
        key = path.resolve()
        if key in seen or not path.exists():
            continue
        seen.add(key)
        paths.append(path)
    return paths


def already_scored_loss(path: Path, val_tokens: int, out_dir: Path | None = None) -> Path | None:
    """Return the CSV that already has this checkpoint, or None."""
    key = _ckpt_key(path)
    for csv_path in _eval_csv_paths("loss_eval.csv", out_dir):
        for row in _read_csv_rows(csv_path):
            if _ckpt_key(Path(row["checkpoint"])) == key and int(row["val_tokens"]) == val_tokens:
                return csv_path
    return None


def already_scored_code(
    path: Path,
    *,
    tasks: str,
    samples: int,
    temperature: float,
    top_k: int,
    n_tasks: int,
    out_dir: Path | None = None,
) -> Path | None:
    key = _ckpt_key(path)
    for csv_path in _eval_csv_paths("code_eval.csv", out_dir):
        for row in _read_csv_rows(csv_path):
            if _ckpt_key(Path(row["checkpoint"])) != key:
                continue
            if row.get("tasks") != tasks:
                continue
            if int(row.get("samples", -1)) != samples:
                continue
            if abs(float(row.get("temperature", -1)) - temperature) > 1e-9:
                continue
            if int(row.get("top_k", -1)) != top_k:
                continue
            if int(row.get("n_tasks", -1)) != n_tasks:
                continue
            return csv_path
    return None


def resolve_out_dir(path: Path, out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir
    parent = path.resolve().parent
    # Branch dirs: checkpoints/cool_18k/latest.pt -> runs/cool_18k
    if parent.parent == (PROJECT_ROOT / "checkpoints").resolve() and parent.name != "checkpoints":
        candidate = FALLBACK_DIR / parent.name
        if candidate.is_dir():
            return candidate
    run_dir = checkpoint_run_dir(path)
    return run_dir if run_dir is not None else FALLBACK_DIR


@dataclass
class LossData:
    val_batches: list[tuple[torch.Tensor, torch.Tensor]]
    train_slice_batches: int  # leading batches that equal the trainer's eval slice
    source_windows: dict[str, torch.Tensor]


def build_loss_data(tokenizer: MorkTokenizer, val_tokens: int) -> LossData:
    cfg = TrainConfig()
    batch_tokens = LOSS_BATCH * cfg.seq_len
    loader = DataLoaderLite(
        B=LOSS_BATCH,
        T=cfg.seq_len,
        split="val",
        data_dir=cfg.data_dir,
        shuffle_chunk_tokens=cfg.shuffle_chunk_tokens,
        fixed_shuffle=True,
        shuffle_seed=cfg.seed,
    )
    n_batches = max(1, val_tokens // batch_tokens)
    val_batches = [loader.next_batch() for _ in range(n_batches)]
    del loader
    train_slice = cfg.eval_tokens // batch_tokens if cfg.eval_tokens % batch_tokens == 0 else 0
    if train_slice > n_batches:
        train_slice = 0

    end_id = tokenizer.special_token_id("<|end|>")
    windows: dict[str, torch.Tensor] = {}
    for source in SOURCES:
        path = RAW_DIR / f"{source}_val.jsonl"
        meta_path = RAW_DIR / f"{source}_val.meta.json"
        if not path.exists():
            print(f"loss: {path} missing, skipping {source}")
            continue
        n_docs = 0
        if meta_path.exists():
            n_docs = int(json.loads(meta_path.read_text()).get("docs", 0))
        if n_docs <= 0:
            with path.open(encoding="utf-8") as f:
                n_docs = sum(1 for _ in f)
        stride = max(1, n_docs // SOURCE_DOCS)
        stream: list[int] = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % stride == 0:
                    ids = tokenizer.encode(json.loads(line)["text"])[:SOURCE_MAX_DOC_TOKENS]
                    stream.extend(ids + [end_id])
        width = cfg.seq_len + 1
        n = min(SOURCE_WINDOWS, len(stream) // width)
        windows[source] = torch.tensor(stream[: n * width], dtype=torch.long).view(n, width)
    return LossData(val_batches, train_slice, windows)


@torch.no_grad()
def evaluate_loss(model: GPT, data: LossData, device: str) -> dict[str, float]:
    def batch_loss(x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x.to(device), y.to(device))
        return float(loss)

    losses = [batch_loss(x, y) for x, y in data.val_batches]
    result = {"val_loss": sum(losses) / len(losses)}
    if data.train_slice_batches:
        head = losses[: data.train_slice_batches]
        result["val_train_slice"] = sum(head) / len(head)
    for source, arr in data.source_windows.items():
        total = count = 0.0
        for i in range(0, len(arr), LOSS_BATCH):
            batch = arr[i : i + LOSS_BATCH]
            total += batch_loss(batch[:, :-1], batch[:, 1:]) * len(batch)
            count += len(batch)
        result[source] = total / count
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python function benchmark for Mork-2 checkpoints.")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        nargs="+",
        default=[DEFAULT_CHECKPOINT],
        help="One or more checkpoint files (default: checkpoints/latest.pt).",
    )
    parser.add_argument(
        "--tasks",
        choices=["loss", "smoke", "mid", "mbpp"],
        nargs="+",
        default=["smoke"],
        help="Task sets to run, e.g. --tasks loss smoke mid (default: smoke).",
    )
    parser.add_argument(
        "--val-tokens",
        type=int,
        default=4_096_000,
        help="Mixed val tokens for --tasks loss (default: 4,096,000).",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=None,
        help="Sampled completions per code task, besides greedy (default: smoke 10, mid 5, mbpp 5).",
    )
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Default: smoke 128, mid 160, mbpp 160.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N code tasks.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for loss_eval.csv / code_eval.csv (default: the checkpoint's run dir).",
    )
    parser.add_argument("--details", type=Path, default=None, help="Also write per-task JSON here.")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--verbose", "-v", action="store_true", help="Print one line per task.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel test subprocesses (default: min(8, CPUs)).",
    )
    parser.add_argument(
        "--format",
        choices=["completion", "instruct"],
        default="completion",
        help=(
            "completion continues the function stub. instruct asks for the "
            "whole function after <|assistant|> and records smoke-instruct, "
            "mid-instruct, mbpp-instruct (default: completion)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if this checkpoint/task is already in the output CSV.",
    )
    args = parser.parse_args(argv)
    args.tasks = list(dict.fromkeys(args.tasks))
    return args


def code_task_label(name: str, args: argparse.Namespace) -> str:
    label = f"{name}-instruct" if args.format == "instruct" else name
    if args.limit is not None:
        label = f"{label}[:{args.limit}]"
    return label


def run_code_tasks(
    model: GPT,
    tokenizer: MorkTokenizer,
    name: str,
    tasks: list[Task],
    args: argparse.Namespace,
    path: Path,
    step: int,
    out_dir: Path,
) -> dict:
    samples = args.samples if args.samples is not None else DEFAULT_SAMPLES[name]
    max_new = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[name]
    t0 = time.time()
    summary, per_task = evaluate_tasks(
        model,
        tokenizer,
        tasks,
        samples=samples,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=max_new,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
        workers=args.workers,
        format=args.format,
    )
    elapsed = time.time() - t0
    n = summary["n_tasks"]
    label = code_task_label(name, args)
    print(
        f"  {label}: greedy {summary['greedy_pass']}/{n} "
        f"(syntax {summary['greedy_syntax']}/{n}), "
        f"sampled pass@1 {summary['sample_pass_at_1']:.3f}, "
        f"pass@{samples} {summary['tasks_solved']}/{n}, "
        f"sample syntax {summary['sample_syntax_rate']:.2f} [{elapsed:.0f}s]",
        flush=True,
    )
    append_csv(
        out_dir / "code_eval.csv",
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(path),
            "step": step,
            "tasks": label,
            "samples": samples,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "sample_pass_at_1": f"{summary['sample_pass_at_1']:.4f}",
            "sample_syntax_rate": f"{summary['sample_syntax_rate']:.4f}",
            "elapsed_s": f"{elapsed:.1f}",
            **{k: summary[k] for k in ("n_tasks", "greedy_pass", "greedy_syntax", "tasks_solved")},
        },
        CSV_FIELDS,
    )
    return {"summary": summary, "per_task": per_task}


def run_loss_task(
    model: GPT, data: LossData, args: argparse.Namespace, path: Path, step: int, out_dir: Path
) -> dict:
    t0 = time.time()
    result = evaluate_loss(model, data, args.device)
    elapsed = time.time() - t0
    val_tokens = len(data.val_batches) * data.val_batches[0][0].numel()
    parts = [f"val {result['val_loss']:.4f} ({val_tokens:,} tokens)"]
    if "val_train_slice" in result:
        parts.append(f"trainer slice {result['val_train_slice']:.4f}")
    parts += [f"{s} {result[s]:.4f}" for s in SOURCES if s in result]
    print(f"  loss: {', '.join(parts)} [{elapsed:.0f}s]", flush=True)
    append_csv(
        out_dir / "loss_eval.csv",
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(path),
            "step": step,
            "val_tokens": val_tokens,
            "elapsed_s": f"{elapsed:.1f}",
            **{k: f"{v:.6f}" for k, v in result.items()},
        },
        LOSS_CSV_FIELDS,
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for path in args.checkpoint:
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")

    torch.set_float32_matmul_precision("high")
    code_sets: dict[str, list[Task]] = {}
    for name in args.tasks:
        if name == "loss":
            continue
        tasks = {"smoke": smoke_tasks, "mid": mid_tasks, "mbpp": mbpp_tasks}[name]()
        code_sets[name] = tasks[: args.limit] if args.limit is not None else tasks
        samples = args.samples if args.samples is not None else DEFAULT_SAMPLES[name]
        print(
            f"{code_task_label(name, args)}: {len(code_sets[name])} tasks, "
            f"greedy + {samples} samples "
            f"(temperature {args.temperature}, top-k {args.top_k})"
        )

    work: list[tuple[Path, list[str]]] = []
    for path in args.checkpoint:
        pending: list[str] = []
        for name in args.tasks:
            if args.force:
                pending.append(name)
                continue
            if name == "loss":
                hit = already_scored_loss(path, args.val_tokens, args.out_dir)
                if hit is not None:
                    print(f"skip {path} loss (already in {hit})")
                else:
                    pending.append(name)
                continue
            samples = args.samples if args.samples is not None else DEFAULT_SAMPLES[name]
            tasks_label = code_task_label(name, args)
            hit = already_scored_code(
                path,
                tasks=tasks_label,
                samples=samples,
                temperature=args.temperature,
                top_k=args.top_k,
                n_tasks=len(code_sets[name]),
                out_dir=args.out_dir,
            )
            if hit is not None:
                print(f"skip {path} {tasks_label} (already in {hit})")
            else:
                pending.append(name)
        if pending:
            work.append((path, pending))
    if not work:
        print("nothing to do")
        return

    tokenizer = MorkTokenizer.load(args.tokenizer)
    need_loss = any("loss" in pending for _, pending in work)
    loss_data = build_loss_data(tokenizer, args.val_tokens) if need_loss else None

    details: dict[str, dict] = {}
    for path, pending in work:
        out_dir = resolve_out_dir(path, args.out_dir)
        model, _, step, _ = load_model_for_inference(path, device=args.device)
        print(f"{path} (step {step}) -> {out_dir}", flush=True)
        details[str(path)] = {"step": step}
        for name in pending:
            if name == "loss":
                assert loss_data is not None
                details[str(path)][name] = run_loss_task(model, loss_data, args, path, step, out_dir)
            else:
                details[str(path)][name] = run_code_tasks(
                    model, tokenizer, name, code_sets[name], args, path, step, out_dir
                )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if args.details is not None:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        args.details.write_text(json.dumps(details, indent=1), encoding="utf-8")
        print(f"wrote {args.details}")


if __name__ == "__main__":
    main()

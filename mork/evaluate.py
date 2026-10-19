"""Executable Python function benchmark for Mork-2 checkpoints.

The model continues a function header and docstring. The body is cut at the
first unindented line, then run with the task's asserts in a subprocess
(``python -I``, temp cwd, 5 s timeout). This is isolation from the eval
process, not a security sandbox.

Task sets:
  smoke  20 hand-written beginner functions, 10 samples each by default.
  mbpp   MBPP sanitized test split (257 tasks), 5 samples each by default.
         Evaluation only; never put MBPP into training data.

Each checkpoint appends one row to ``<run_dir>/code_eval.csv`` (the run that
wrote the checkpoint), or to ``--out``.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from mork.checkpoint import load_model_for_inference
from mork.config import PROJECT_ROOT
from mork.generate import DEFAULT_TOKENIZER_PATH, toplevel_cut
from mork.model import GPT
from mork.tokenizer import MorkTokenizer

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "latest.pt"
FALLBACK_CSV = PROJECT_ROOT / "runs" / "code_eval.csv"
DEFAULT_SAMPLES = {"smoke": 10, "mbpp": 5}

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


def run_program(program: str, workdir: str) -> tuple[bool, bool]:
    """Returns (compiles, passes)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(program, "<gen>", "exec")
    except Exception:
        return False, False
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", program],
            cwd=workdir,
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
) -> list[str]:
    """``n`` completions in one batch; ``temperature == 0`` is greedy."""
    block = model.config.block_size
    end_id = tokenizer.special_token_id("<|end|>")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    idx = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    rows = list(range(n))  # original sample index of each row still in idx
    done: dict[int, list[int]] = {}
    for step in range(max_new_tokens):
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(idx[:, -block:])
        logits = logits[:, -1, :].float()
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
                if end_id in out or toplevel_cut(tokenizer.decode(out)) is not None:
                    done[rows[j]] = out
                else:
                    keep.append(j)
            if not keep:
                break
            if len(keep) < len(rows):
                idx = idx[keep]
                rows = [rows[j] for j in keep]
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
) -> tuple[dict, list[dict]]:
    per_task = []
    with tempfile.TemporaryDirectory() as workdir:
        for i, task in enumerate(tasks):
            ids = tokenizer.encode(task.prompt)
            common = dict(max_new_tokens=max_new_tokens, top_k=top_k, device=device)
            greedy = sample(model, tokenizer, ids, n=1, temperature=0.0, seed=seed, **common)[0]
            greedy_body = extract_body(greedy)
            g_syntax, g_pass = run_program(
                task.setup + task.prompt + greedy_body + "\n\n" + task.tests, workdir
            )
            s_syntax = s_pass = 0
            if samples > 0:
                outs = sample(
                    model, tokenizer, ids, n=samples, temperature=temperature,
                    seed=seed + 1000 + i, **common,
                )
                for out in outs:
                    ok_syntax, ok = run_program(
                        task.setup + task.prompt + extract_body(out) + "\n\n" + task.tests,
                        workdir,
                    )
                    s_syntax += ok_syntax
                    s_pass += ok
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
            elif (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(tasks)} tasks", flush=True)

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


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


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
    parser.add_argument("--tasks", choices=["smoke", "mbpp"], default="smoke")
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=None,
        help="Sampled completions per task, besides greedy (default: smoke 10, mbpp 5).",
    )
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Default: smoke 128, mbpp 160.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N tasks.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV to append to (default: <checkpoint run_dir>/code_eval.csv).",
    )
    parser.add_argument("--details", type=Path, default=None, help="Also write per-task JSON here.")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--verbose", "-v", action="store_true", help="Print one line per task.")
    args = parser.parse_args(argv)
    if args.samples is None:
        args.samples = DEFAULT_SAMPLES[args.tasks]
    if args.max_new_tokens is None:
        args.max_new_tokens = 128 if args.tasks == "smoke" else 160
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for path in args.checkpoint:
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")

    torch.set_float32_matmul_precision("high")
    tokenizer = MorkTokenizer.load(args.tokenizer)
    tasks = smoke_tasks() if args.tasks == "smoke" else mbpp_tasks()
    if args.limit is not None:
        tasks = tasks[: args.limit]
    print(
        f"{args.tasks}: {len(tasks)} tasks, greedy + {args.samples} samples "
        f"(temperature {args.temperature}, top-k {args.top_k})"
    )

    details: dict[str, dict] = {}
    for path in args.checkpoint:
        t0 = time.time()
        model, _, step, _ = load_model_for_inference(path, device=args.device)
        summary, per_task = evaluate_tasks(
            model,
            tokenizer,
            tasks,
            samples=args.samples,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            seed=args.seed,
            verbose=args.verbose,
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.time() - t0

        n = summary["n_tasks"]
        print(
            f"{path} (step {step}): greedy {summary['greedy_pass']}/{n} "
            f"(syntax {summary['greedy_syntax']}/{n}), "
            f"sampled pass@1 {summary['sample_pass_at_1']:.3f}, "
            f"pass@{args.samples} {summary['tasks_solved']}/{n}, "
            f"sample syntax {summary['sample_syntax_rate']:.2f} [{elapsed:.0f}s]",
            flush=True,
        )

        out = args.out
        if out is None:
            run_dir = checkpoint_run_dir(path)
            out = run_dir / "code_eval.csv" if run_dir is not None else FALLBACK_CSV
        append_csv(
            out,
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(path),
                "step": step,
                "tasks": args.tasks if args.limit is None else f"{args.tasks}[:{args.limit}]",
                "samples": args.samples,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "sample_pass_at_1": f"{summary['sample_pass_at_1']:.4f}",
                "sample_syntax_rate": f"{summary['sample_syntax_rate']:.4f}",
                "elapsed_s": f"{elapsed:.1f}",
                **{k: summary[k] for k in ("n_tasks", "greedy_pass", "greedy_syntax", "tasks_solved")},
            },
        )
        print(f"appended to {out}")
        details[str(path)] = {"step": step, "summary": summary, "per_task": per_task}

    if args.details is not None:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        args.details.write_text(json.dumps(details, indent=1), encoding="utf-8")
        print(f"wrote {args.details}")


if __name__ == "__main__":
    main()

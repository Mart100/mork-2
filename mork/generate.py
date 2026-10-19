import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from mork.checkpoint import (
    best_val_checkpoint_path,
    latest_checkpoint_path,
    load_model_for_inference,
)
from mork.config import PROJECT_ROOT
from mork.model import GPT
from mork.tokenizer import MorkTokenizer

DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a Mork-2 checkpoint.")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="Checkpoint file (default: checkpoints/latest.pt, else best_val.pt).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="Tokenizer JSON file.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="The",
        help=(
            "Text prompt to continue from. End it on a word or symbol, not a "
            "space or newline: the tokenizer merges those with the next token."
        ),
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=3,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="Maximum new tokens per sample. Stops early on <|end|>.",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=None,
        help=(
            "Sampling temperature (higher = more random). "
            "Default: 0.3 with --stop-at-toplevel, else 0.7."
        ),
    )
    parser.add_argument(
        "--stop-at-toplevel",
        action="store_true",
        help=(
            "Stop at the first unindented non-empty line after the prompt, "
            "e.g. after one Python function body."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (0 = disabled).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (incremented per sample).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (cuda or cpu).",
    )
    args = parser.parse_args(argv)
    if args.temperature is None:
        args.temperature = 0.3 if args.stop_at_toplevel else 0.7
    return args


def resolve_checkpoint(path: Path | None) -> Path:
    if path is not None:
        return path
    latest = latest_checkpoint_path(DEFAULT_CHECKPOINT_DIR)
    if latest is not None:
        return latest
    best = best_val_checkpoint_path(DEFAULT_CHECKPOINT_DIR)
    if best is not None:
        return best
    return DEFAULT_CHECKPOINT_DIR / "latest.pt"


def toplevel_cut(generated: str) -> int | None:
    """Index of the newline before the first unindented non-empty line, if any.

    The first line continues the prompt's last line, so it is never a cut point.
    """
    pos = generated.find("\n")
    while pos != -1:
        nxt = pos + 1
        if nxt < len(generated) and not generated[nxt].isspace():
            return pos
        pos = generated.find("\n", nxt)
    return None


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: MorkTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    stop_at_toplevel: bool = False,
) -> tuple[str, dict[str, float | int]]:
    tokens = tokenizer.encode(prompt)
    if not tokens:
        raise ValueError("prompt encoded to zero tokens")
    prompt_len = len(tokens)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    end_id = tokenizer.special_token_id("<|end|>")

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        context = idx[:, -model.config.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            cutoff = torch.topk(logits, k).values[:, -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

        if next_id.item() == end_id:
            break
        if stop_at_toplevel:
            generated = tokenizer.decode(idx[0, prompt_len:].tolist())
            if toplevel_cut(generated) is not None:
                break

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    new_tokens = int(idx.size(1) - prompt_len)
    tokens_per_sec = new_tokens / elapsed_s if elapsed_s > 0 else 0.0
    ids = idx[0].tolist()
    prompt_text = tokenizer.decode(ids[:prompt_len])
    generated_text = tokenizer.decode(ids[prompt_len:])
    full_text = tokenizer.decode(ids)
    if stop_at_toplevel:
        cut = toplevel_cut(generated_text)
        if cut is not None:
            generated_text = generated_text[:cut]
            full_text = prompt_text + generated_text
    stats = {
        "prompt_tokens": prompt_len,
        "new_tokens": new_tokens,
        "elapsed_ms": elapsed_s * 1000.0,
        "tokens_per_sec": tokens_per_sec,
        "prompt_text": prompt_text,
        "generated_text": generated_text.rstrip(),
    }
    return full_text.rstrip(), stats


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)

    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if not args.tokenizer.exists():
        raise SystemExit(f"tokenizer not found: {args.tokenizer}")

    tokenizer = MorkTokenizer.load(args.tokenizer)
    model, _, step, loss = load_model_for_inference(checkpoint, device=args.device)

    print(f"loaded {checkpoint} (step {step}, loss {loss:.4f})")
    print(f"prompt: {args.prompt!r}  temperature: {args.temperature}")
    if args.prompt and args.prompt[-1] in " \t\n":
        print(
            "warning: the prompt ends in whitespace. The tokenizer merges a "
            "trailing space or newline with the next word or indentation, so "
            "this puts the model off-distribution. End on a word or symbol "
            "instead (e.g. after '\"\"\"' or ':')."
        )
    print()

    for i in range(args.samples):
        torch.manual_seed(args.seed + i)
        if args.device == "cuda":
            torch.cuda.manual_seed(args.seed + i)

        text, stats = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
            stop_at_toplevel=args.stop_at_toplevel,
        )
        print(f"--- sample {i + 1} ---")
        print(text)
        print(
            f"[inference] {stats['new_tokens']} new tokens | "
            f"{stats['elapsed_ms']:.1f} ms | "
            f"{stats['tokens_per_sec']:.1f} tokens/sec"
        )
        print()


if __name__ == "__main__":
    main()

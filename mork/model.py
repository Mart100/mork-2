import torch
import torch.nn as nn
from torch.nn import functional as F

from mork.config import ModelConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pair dims as (a, b) -> (-b, a) for RoPE (LLaMA-style)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key. q,k: (B, H, T, D); cos/sin: (T, D)."""
    cos = cos.to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for rotary position embeddings."""

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE head dim must be even, got {dim}"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim) for rotate_half
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class KVCache:
    """Per-layer keys/values (after RoPE) for incremental decoding.

    Buffers are allocated on first use with the dtype attention produces, so
    autocast and fp32 inference both work. Positions are absolute: ``pos`` is
    the number of tokens already cached, and can't exceed ``max_len``.
    """

    def __init__(self, n_layer: int, max_len: int):
        self.n_layer = n_layer
        self.max_len = max_len
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer
        self.pos = 0

    def reset(self) -> None:
        self.k = [None] * self.n_layer
        self.v = [None] * self.n_layer
        self.pos = 0

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, T, D = k.shape
        end = self.pos + T
        if end > self.max_len:
            raise ValueError(f"KV cache full: {end} > {self.max_len}")
        if self.k[layer] is None:
            self.k[layer] = k.new_zeros(B, H, self.max_len, D)
            self.v[layer] = v.new_zeros(B, H, self.max_len, D)
        k_buf, v_buf = self.k[layer], self.v[layer]
        assert k_buf is not None and v_buf is not None
        k_buf[:, :, self.pos : end] = k
        v_buf[:, :, self.pos : end] = v
        return k_buf[:, :, :end], v_buf[:, :, :end]

    def keep_rows(self, rows: torch.Tensor) -> None:
        """Drop batch rows (e.g. finished samples); ``rows`` indexes the kept ones."""
        for i in range(self.n_layer):
            if self.k[i] is not None:
                self.k[i] = self.k[i].index_select(0, rows)
                self.v[i] = self.v[i].index_select(0, rows)


class RMSNorm(nn.Module):
    """LLaMA-style RMSNorm (no mean center, scale-only, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore[attr-defined]
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is None:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
        else:
            start = kv_cache.pos
            k, v = kv_cache.update(layer, k, v)
            if T == 1:
                y = F.scaled_dot_product_attention(q, k, v)
            elif start == 0:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                # New query i sits at absolute position start + i.
                rows = torch.arange(T, device=x.device).unsqueeze(1) + start
                cols = torch.arange(start + T, device=x.device).unsqueeze(0)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=cols <= rows)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """SwiGLU feed-forward; hidden ≈ (8/3)d so params stay near the old GELU-4d MLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # 3 * d * h ≈ 8 * d^2  =>  h ≈ (8/3) d; round to 64 for GPU-friendly shapes.
        hidden = 64 * round((8 * config.n_embd) / (3 * 64))
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden, bias=False)  # gate + up fused
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore[attr-defined]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.c_fc(x).chunk(2, dim=-1)
        return self.dropout(self.c_proj(F.silu(gate) * up))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cos, sin, kv_cache, layer)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(
            head_dim, config.block_size, base=config.rope_theta
        )
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        ignore_index: int = -100,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        start = 0 if kv_cache is None else kv_cache.pos
        assert start + T <= self.config.block_size, (
            f"Cannot forward sequence of length {start + T}, "
            f"block size is only {self.config.block_size}"
        )
        if kv_cache is None:
            cos, sin = self.rotary(T)
        else:
            cos = self.rotary.cos_cached[start : start + T]
            sin = self.rotary.sin_cached[start : start + T]

        x = self.drop(self.wte(idx))
        for i, block in enumerate(self.h):
            x = block(x, cos, sin, kv_cache, i)
        if kv_cache is not None:
            kv_cache.pos += T
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # ignore_index lets a later SFT stage mask prompt tokens. Pretrain
            # targets are real token ids, so the default changes nothing.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    def new_kv_cache(self) -> KVCache:
        return KVCache(self.config.n_layer, self.config.block_size)

    @torch.no_grad()
    def next_logits(self, ids: torch.Tensor, kv_cache: KVCache, n_new: int) -> torch.Tensor:
        """Logits for the token after ``ids`` (B, L); its last ``n_new`` tokens aren't cached yet.

        Past block_size the cache is rebuilt from the last 3/4 of the context,
        since positions beyond block_size were never trained.
        """
        block = self.config.block_size
        if kv_cache.pos > 0 and kv_cache.pos + n_new <= block:
            logits, _ = self(ids[:, -n_new:], kv_cache=kv_cache)
        else:
            keep = min(ids.size(1), block if kv_cache.pos == 0 else (3 * block) // 4)
            kv_cache.reset()
            logits, _ = self(ids[:, -keep:], kv_cache=kv_cache)
        return logits[:, -1, :]

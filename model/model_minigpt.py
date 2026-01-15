
"""MiniGPT ()

Refactors the original MiniGPT implementation to adopt MiniMind-style training/inference details:

- Long-context RoPE (default max_position_embeddings=32768, rope_theta=1e6) with optional YaRN-like scaling.
- GQA/MQA via num_key_value_heads + repeat_kv.
- KV cache (past_key_values) + use_cache for fast autoregressive decoding.
- Optional Flash Attention via torch.scaled_dot_product_attention + attention_mask support.
- MiniMind-style SwiGLU MLP: gate_proj + up_proj + down_proj; default intermediate ~= 8/3*hidden aligned to 64.
- MiniMind-style MoE: routed experts + optional shared experts + auxiliary load-balancing loss.
- Optional logits_to_keep to only return last N logits.
- Optional labels to compute causal cross-entropy loss.
- Pure PyTorch (no HuggingFace dependency), but interface resembles HF causal LM outputs.

Usage:
    cfg = MiniGPTConfig(...)
    model = MiniGPTForCausalLM(cfg)
    out = model(input_ids, attention_mask=..., labels=..., use_cache=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import math
import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------
# Config (expanded to match MiniMind-style knobs)
# ----------------------------
@dataclass
class MiniGPTConfig:
    # Core
    vocab_size: int = 6400
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8  # set < num_attention_heads to enable GQA/MQA
    max_position_embeddings: int = 32768

    # MLP / activation
    intermediate_size: int | None = None
    hidden_act: str = "silu"

    # Norm / dropout / init
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02

    # RoPE
    rope_theta: float = 1_000_000.0
    rope_scaling: Optional[dict] = None  # {"type":"yarn","factor":..., "beta_fast":..., "beta_slow":..., "original_max_position_embeddings":...}

    # MoE (MiniMind-style)
    use_moe: bool = False
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    n_shared_experts: int = 0
    aux_loss_alpha: float = 0.01
    seq_aux: bool = True
    norm_topk_prob: bool = True

    # Attention impl
    flash_attn: bool = True  # uses torch.scaled_dot_product_attention when available

    # Tie embeddings
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads must be <= num_attention_heads.")
        if self.use_moe:
            if self.n_routed_experts < 1:
                raise ValueError("n_routed_experts must be >= 1 when use_moe=True.")
            if self.num_experts_per_tok < 1:
                raise ValueError("num_experts_per_tok must be >= 1 when use_moe=True.")
            if self.num_experts_per_tok > self.n_routed_experts:
                raise ValueError("num_experts_per_tok must be <= n_routed_experts.")
            if self.n_shared_experts < 0:
                raise ValueError("n_shared_experts must be >= 0.")


# ----------------------------
# Outputs
# ----------------------------
@dataclass
class MiniGPTCausalLMOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    aux_loss: Optional[torch.Tensor] = None


# ----------------------------
# Utilities
# ----------------------------
_ACT: dict[str, Any] = {
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}

def _get_act(name: str):
    if name not in _ACT:
        raise ValueError(f"Unsupported hidden_act={name}. Supported: {sorted(_ACT)}")
    return _ACT[name]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat kv heads for GQA.

    x: (B, n_kv_heads, S, D) -> (B, n_kv_heads*n_rep, S, D)
    """
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return x.reshape(b, h * n_rep, s, d)


def yarn_find_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int):
    low = math.floor(dim * math.log(max_position_embeddings / low_rot) / (2 * math.log(base)))
    high = math.ceil(dim * math.log(max_position_embeddings / high_rot) / (2 * math.log(base)))
    return max(low, 0), min(high, dim // 2)


def yarn_linear_ramp_mask(low: int, high: int, dim_half: int) -> torch.Tensor:
    if low == high:
        high = low + 1
    ramp = (torch.arange(dim_half).float() - low) / (high - low)
    return torch.clamp(ramp, 0, 1)


@torch.no_grad()
def precompute_rope_cos_sin(
    dim: int,
    max_pos: int,
    base: float,
    rope_scaling: Optional[dict] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables.

    Returns:
      cos: (max_pos, dim)
      sin: (max_pos, dim)
    """
    device = device or torch.device("cpu")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # YaRN-like scaling (same intent as MiniMind)
    if rope_scaling and rope_scaling.get("type", "").lower() == "yarn":
        factor = float(rope_scaling["factor"])
        beta_fast = float(rope_scaling.get("beta_fast", 32))
        beta_slow = float(rope_scaling.get("beta_slow", 1))
        orig_max = int(rope_scaling.get("original_max_position_embeddings", max_pos))

        low, high = yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max)
        inv_freq_extrap = inv_freq / factor
        inv_freq_interp = inv_freq
        mask = 1 - yarn_linear_ramp_mask(low, high, dim // 2).to(device)
        inv_freq = inv_freq_interp * (1 - mask) + inv_freq_extrap * mask

    t = torch.arange(max_pos, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_pos, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)        # (max_pos, dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k.

    q,k: (B, H, S, D)
    cos,sin: (S, D)
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rot = torch.stack((-q2, q1), dim=-1).flatten(-2)
    k_rot = torch.stack((-k2, k1), dim=-1).flatten(-2)
    q = q * cos + q_rot * sin
    k = k * cos + k_rot * sin
    return q, k


# ----------------------------
# Attention (MiniMind-style: GQA + KV cache + attention_mask + optional flash)
# ----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        hs = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if hs % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = hs // self.n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")

        self.q_proj = nn.Linear(hs, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hs, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hs, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hs, hs, bias=False)

        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.flash_attn = bool(config.flash_attn) and hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward.

        x: (B,S,C)
        attention_mask: (B, S_total) keys mask where 1=keep, 0=pad. If provided as (B,S) and cache exists,
                        past prefix ones are prepended automatically.
        past_key_value: (k,v) each (B, n_kv_heads, S_past, head_dim)
        """
        b, s, c = x.shape

        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)      # (B,H,S,D)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B,Hkv,S,D)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        past_len = 0
        if past_key_value is not None:
            pk, pv = past_key_value
            past_len = pk.shape[2]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        present = (k, v) if use_cache else None

        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)
        s_total = k_rep.shape[2]

        # normalize attention_mask to (B, S_total)
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be 2D (B, S_total) or (B, S).")
            if attention_mask.shape[1] == s and past_len > 0:
                prefix = torch.ones((b, past_len), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix, attention_mask], dim=1)
            if attention_mask.shape[1] != s_total:
                raise ValueError(f"attention_mask length mismatch: got {attention_mask.shape[1]}, expected {s_total}")

        if self.flash_attn:
            if attention_mask is None and past_len == 0:
                out = F.scaled_dot_product_attention(
                    q, k_rep, v_rep,
                    attn_mask=None,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                causal_mask = torch.triu(
                    torch.ones((s, s_total), device=x.device, dtype=torch.bool),
                    diagonal=1 + past_len,
                )
                attn_bias = torch.zeros((s, s_total), device=x.device, dtype=q.dtype)
                attn_bias = attn_bias.masked_fill(causal_mask, float("-inf"))
                if attention_mask is not None:
                    pad_mask = ~attention_mask.to(torch.bool)
                    pad_bias = torch.zeros((b, 1, 1, s_total), device=x.device, dtype=q.dtype)
                    pad_bias = pad_bias.masked_fill(pad_mask[:, None, None, :], float("-inf"))
                    attn_mask = attn_bias[None, None, :, :] + pad_bias
                else:
                    attn_mask = attn_bias[None, None, :, :]
                out = F.scaled_dot_product_attention(
                    q, k_rep, v_rep,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=False,
                )
        else:
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,S,S_total)

            causal = torch.full((s, s_total), float("-inf"), device=x.device, dtype=scores.dtype)
            causal = torch.triu(causal, diagonal=1 + past_len)
            scores = scores + causal[None, None, :, :]

            if attention_mask is not None:
                pad_mask = ~attention_mask.to(torch.bool)
                scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v_rep)

        out = out.transpose(1, 2).contiguous().view(b, s, c)
        out = self.o_proj(out)
        return out, present


# ----------------------------
# MLP (MiniMind-style)
# ----------------------------
def _default_intermediate(hidden_size: int) -> int:
    x = int(hidden_size * 8 / 3)
    return (x + 63) // 64 * 64


class MLP(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = config.intermediate_size if config.intermediate_size is not None else _default_intermediate(hidden)
        act = _get_act(config.hidden_act)

        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act = act
        self.drop = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        return self.drop(x)


# ----------------------------
# MoE (MiniMind-style)
# ----------------------------
class MoEGate(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.n_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.aux_loss_alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.gate = nn.Linear(config.hidden_size, self.n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        b, s, _ = x.shape
        logits = self.gate(x)                   # (B,S,E)
        scores = torch.softmax(logits, dim=-1)  # (B,S,E)

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)  # (B,S,K)
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-9)

        aux_loss = None
        if self.training and self.aux_loss_alpha > 0:
            if self.seq_aux:
                probs = scores.sum(dim=1)  # (B,E)
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
                top1 = topk_idx[..., 0]    # (B,S)
                cnt = torch.zeros((b, self.n_experts), device=x.device, dtype=scores.dtype)
                cnt.scatter_add_(1, top1, torch.ones_like(top1, dtype=scores.dtype))
                cnt = cnt / (cnt.sum(dim=-1, keepdim=True) + 1e-9)
                aux_loss = (probs * cnt).sum(dim=-1).mean() * (self.n_experts * self.aux_loss_alpha)
            else:
                probs = scores.reshape(-1, self.n_experts)
                top1 = topk_idx[..., 0].reshape(-1)
                cnt = torch.zeros((self.n_experts,), device=x.device, dtype=scores.dtype)
                cnt.scatter_add_(0, top1, torch.ones_like(top1, dtype=scores.dtype))
                cnt = cnt / (cnt.sum() + 1e-9)
                probs_mean = probs.mean(dim=0)
                aux_loss = (probs_mean * cnt).sum() * (self.n_experts * self.aux_loss_alpha)

        return topk_idx.reshape(-1, self.top_k), topk_weight.reshape(-1, self.top_k), aux_loss


class MoE(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.k = config.num_experts_per_tok
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.n_routed)])
        self.gate = MoEGate(config)

        self.shared = None
        if config.n_shared_experts and config.n_shared_experts > 0:
            shared_cfg = MiniGPTConfig(**{**config.__dict__})
            shared_cfg.intermediate_size = _default_intermediate(config.hidden_size) * int(config.n_shared_experts)
            self.shared = MLP(shared_cfg)

    @torch.no_grad()
    def _moe_infer(self, x_flat: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        t, c = x_flat.shape
        k = topk_idx.shape[1]
        expanded_x = x_flat.repeat_interleave(k, dim=0)  # (T*K, C)
        expanded_w = topk_weight.reshape(-1)             # (T*K,)
        expanded_e = topk_idx.reshape(-1)                # (T*K,)

        order = torch.argsort(expanded_e)
        expanded_x = expanded_x[order]
        expanded_w = expanded_w[order]
        expanded_e = expanded_e[order]

        counts = torch.bincount(expanded_e, minlength=self.n_routed)
        ends = torch.cumsum(counts, dim=0)

        out = torch.zeros_like(expanded_x)
        start = 0
        for eid, end in enumerate(ends.tolist()):
            if end > start:
                out[start:end] = self.experts[eid](expanded_x[start:end])
            start = end

        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(order.numel(), device=order.device)
        out = out[inv_order]  # (T*K, C)
        out = out * expanded_w[:, None]
        out = out.reshape(t, k, c).sum(dim=1)  # (T, C)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, s, c = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)   # (B*S, K)
        x_flat = x.reshape(-1, c)                        # (T, C)

        if self.training:
            k = topk_idx.shape[1]
            expanded = x_flat.repeat_interleave(k, dim=0)  # (T*K, C)
            out = torch.zeros_like(expanded)
            expert_id = topk_idx.reshape(-1)               # (T*K,)

            for eid, expert in enumerate(self.experts):
                mask = (expert_id == eid)
                if mask.any():
                    out[mask] = expert(expanded[mask])

            out = out.view(-1, k, c) * topk_weight.view(-1, k, 1)
            out = out.sum(dim=1)  # (T, C)
        else:
            out = self._moe_infer(x_flat, topk_idx, topk_weight)  # (T, C)

        out = out.view(b, s, c)

        if self.shared is not None:
            out = out + self.shared(x)

        return out, aux_loss


# ----------------------------
# Decoder block
# ----------------------------
class DecoderLayer(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_moe = bool(config.use_moe)
        self.ffn = MoE(config) if self.use_moe else MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        # Self-attn
        h, present = self.attn(
            self.attn_norm(x),
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + h

        # FFN / MoE
        if self.use_moe:
            h, aux = self.ffn(self.ffn_norm(x))
        else:
            h = self.ffn(self.ffn_norm(x))
            aux = None
        x = x + h
        return x, present, aux



# ----------------------------
# Base model
# ----------------------------
class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Precompute RoPE tables as buffers (MiniMind style)
        cos, sin = precompute_rope_cos_sin(
            dim=config.hidden_size // config.num_attention_heads,
            max_pos=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _select_rope(self, start_pos: int, seq_len: int, device: torch.device, dtype: torch.dtype):
        end_pos = start_pos + seq_len
        if end_pos > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence positions [{start_pos}, {end_pos}) exceed max_position_embeddings={self.config.max_position_embeddings}."
            )
        cos = self.rope_cos[start_pos:end_pos].to(device=device, dtype=dtype)
        sin = self.rope_sin[start_pos:end_pos].to(device=device, dtype=dtype)
        return cos, sin

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[torch.Tensor]]:
        """Forward hidden states.

        input_ids: (B,S)
        attention_mask: (B, S_total) keys mask (1=keep,0=pad). If cache is used, S_total should include past+current,
                        or pass (B,S) and we will prepend ones automatically inside attention.
        past_key_values: list length L; each (k,v) with k/v shape (B, n_kv_heads, past_len, head_dim)
        """
        b, s = input_ids.shape
        device = input_ids.device

        x = self.embed_tokens(input_ids)  # (B,S,C)
        dtype = x.dtype

        start_pos = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            start_pos = past_key_values[0][0].shape[2]

        cos, sin = self._select_rope(start_pos, s, device=device, dtype=dtype)

        next_past = [] if use_cache else None
        aux_losses = []

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        elif len(past_key_values) != len(self.layers):
            raise ValueError(
                f"past_key_values length {len(past_key_values)} does not match "
                f"num_hidden_layers {len(self.layers)}."
            )

        for layer, pkv in zip(self.layers, past_key_values):
            x, present, aux = layer(
                x,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                past_key_value=pkv,
                use_cache=use_cache,
            )
            if use_cache:
                next_past.append(present)
            if aux is not None:
                aux_losses.append(aux)

        x = self.norm(x)

        aux_loss = None
        if aux_losses:
            aux_loss = torch.stack(aux_losses).sum()

        return x, next_past, aux_loss


# ----------------------------
# Causal LM wrapper
# ----------------------------
class MiniGPTForCausalLM(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.model = MiniGPT(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Optional[int] = None,
    ) -> MiniGPTCausalLMOutput:
        hidden, next_past, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if logits_to_keep is not None:
            hidden_for_logits = hidden[:, -logits_to_keep:, :]
        else:
            hidden_for_logits = hidden

        logits = self.lm_head(hidden_for_logits)

        loss = None
        if labels is not None:
            if logits.size(1) < 2:
                raise ValueError("Need at least 2 logits to compute loss.")
            # Causal LM loss: predict next token
            # Align with logits we produced:
            if logits_to_keep is not None:
                # Only last K logits correspond to last K positions
                # We still do next-token prediction within that window.
                # Need matching label slice: positions for those logits.
                label_slice = labels[:, -logits_to_keep:]
                # shift
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = label_slice[:, 1:].contiguous()
            else:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        return MiniGPTCausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=next_past,
            aux_loss=aux_loss,
        )

    @torch.no_grad()
    def generate_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """One autoregressive step: returns next token and updated cache."""
        out = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = out.logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            thresh = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token, out.past_key_values


__all__ = [
    "MiniGPTConfig",
    "MiniGPTCausalLMOutput",
    "MiniGPT",
    "MiniGPTForCausalLM",
]

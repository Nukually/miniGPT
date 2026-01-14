from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MiniGPTConfig:
    vocab_size: int = 6400
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    max_position_embeddings: int = 512
    intermediate_size: int | None = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    moe_num_experts: int = 1
    moe_top_k: int = 1


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(1), persistent=False)
        self.register_buffer("sin_cached", torch.empty(1), persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def _build_cache(self, seq_len: int, device, dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos().to(dtype=dtype)
        self.sin_cached = emb.sin().to(dtype=dtype)
        self.max_seq_len_cached = seq_len

    def forward(self, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.max_position_embeddings}."
            )
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len, device, dtype)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        mask = torch.triu(
            torch.ones(config.max_position_embeddings, config.max_position_embeddings),
            diagonal=1,
        ).bool()
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_scores = attn_scores.masked_fill(
            self.mask[:seq_len, :seq_len],
            float("-inf"),
        )
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size or hidden_size * 4
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act(gate) * up)


class MoE(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.num_experts = max(1, config.moe_num_experts)
        self.top_k = max(1, min(config.moe_top_k, self.num_experts))
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        if self.top_k < self.num_experts:
            topk_vals, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)
            masked = torch.zeros_like(gate_probs)
            masked.scatter_(-1, topk_idx, topk_vals)
            gate_probs = masked
            denom = gate_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            gate_probs = gate_probs / denom

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        return torch.einsum("bse,ebsh->bsh", gate_probs, expert_outputs)


class TransformerBlock(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = MoE(config) if config.moe_num_experts > 1 else FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


@dataclass
class MiniGPTOutput:
    logits: torch.Tensor


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> MiniGPTOutput:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return MiniGPTOutput(logits=logits)

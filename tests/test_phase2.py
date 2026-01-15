"""
test_phase2.py

A lightweight functional test suite for the main/core modules in model_minigpt.py.

How to run:
  python test_phase2.py

Notes:
- Runs on CPU by default.
- Uses small configs to keep runtime low.
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import Tuple, List, Optional

import torch



# Resolve project layout:
#   <repo_root>/
#     model/model_minigpt.py
#     test/test_phase2.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_MODEL_DIR = os.path.join(_REPO_ROOT, "model")

# Add model directory to sys.path so `import model_minigpt` works from test folder
for p in (_MODEL_DIR, _REPO_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import model_minigpt as m
except Exception as e:
    raise RuntimeError(
        "Failed to import model_minigpt.py. Make sure test_phase2.py is in the same folder "
        "or that model_minigpt.py is on PYTHONPATH. Original error:\n"
        f"{repr(e)}"
    ) from e


def _seed_all(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _small_config(use_moe: bool = False) -> "m.MiniGPTConfig":
    # Keep shapes tiny so tests run fast on CPU
    cfg = m.MiniGPTConfig(
        vocab_size=128,
        hidden_size=32,              # must be divisible by num_attention_heads
        num_attention_heads=4,       # head_dim=8 (even, required by rotary rotate_half)
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=64,
        dropout=0.0,
        flash_attn=False,
        use_moe=use_moe,
        # MoE specific
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        aux_loss_alpha=0.01,
        seq_aux=True,
        norm_topk_prob=True,
    )
    return cfg


class TestMiniGPTCoreModules(unittest.TestCase):
    def setUp(self) -> None:
        _seed_all(7)
        self.device = torch.device("cpu")

    def test_config_rope_scaling_toggle(self):
        cfg0 = m.MiniGPTConfig(inference_rope_scaling=False)
        self.assertIsNone(cfg0.rope_scaling)

        cfg1 = m.MiniGPTConfig(inference_rope_scaling=True)
        self.assertIsInstance(cfg1.rope_scaling, dict)
        self.assertIn("factor", cfg1.rope_scaling)

    def test_rmsnorm_basic(self):
        dim = 8
        layer = m.RMSNorm(dim, eps=1e-5).to(self.device)
        x = torch.randn(2, 3, dim, device=self.device)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

        # Compare to manual normalization (up to numerical tolerance)
        with torch.no_grad():
            denom = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + layer.eps)
            y_ref = layer.weight * (x * denom)
            self.assertTrue(torch.allclose(y, y_ref, atol=1e-5, rtol=1e-4))

    def test_precompute_freqs_shapes(self):
        head_dim = 8
        end = 17
        cos, sin = m.precompute_freqs_cis(dim=head_dim, end=end, rope_base=1e6, rope_scaling=None)
        self.assertEqual(cos.shape, (end, head_dim))
        self.assertEqual(sin.shape, (end, head_dim))
        self.assertTrue(torch.isfinite(cos).all())
        self.assertTrue(torch.isfinite(sin).all())

    def test_apply_rotary_pos_emb_shapes(self):
        bsz, seqlen, nheads, head_dim = 2, 5, 4, 8
        q = torch.randn(bsz, seqlen, nheads, head_dim, device=self.device)
        k = torch.randn(bsz, seqlen, nheads, head_dim, device=self.device)
        cos, sin = m.precompute_freqs_cis(dim=head_dim, end=seqlen, rope_base=1e6, rope_scaling=None)
        q2, k2 = m.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        self.assertEqual(q2.shape, q.shape)
        self.assertEqual(k2.shape, k.shape)
        self.assertTrue(torch.isfinite(q2).all())
        self.assertTrue(torch.isfinite(k2).all())

    def test_repeat_kv(self):
        x = torch.randn(2, 3, 2, 8, device=self.device)
        y = m.repeat_kv(x, n_rep=2)
        self.assertEqual(y.shape, (2, 3, 4, 8))
        z = m.repeat_kv(x, n_rep=1)
        self.assertTrue(torch.equal(z, x))

    def test_attention_forward_and_cache(self):
        cfg = _small_config(use_moe=False)
        attn = m.Attention(cfg).to(self.device)

        bsz, seqlen = 2, 4
        x = torch.randn(bsz, seqlen, cfg.hidden_size, device=self.device)

        head_dim = cfg.hidden_size // cfg.num_attention_heads
        cos, sin = m.precompute_freqs_cis(dim=head_dim, end=cfg.max_position_embeddings, rope_base=cfg.rope_theta, rope_scaling=None)
        pos_emb = (cos[:seqlen], sin[:seqlen])

        out, past = attn(x, pos_emb, past_key_value=None, use_cache=True, attention_mask=None)
        self.assertEqual(out.shape, (bsz, seqlen, cfg.hidden_size))
        self.assertIsNotNone(past)
        self.assertEqual(past[0].shape, (bsz, seqlen, attn.n_local_kv_heads, head_dim))
        self.assertEqual(past[1].shape, (bsz, seqlen, attn.n_local_kv_heads, head_dim))

        # Step with cache: feed one more token and ensure kv length grows
        x2 = torch.randn(bsz, 1, cfg.hidden_size, device=self.device)
        pos_emb2 = (cos[seqlen:seqlen + 1], sin[seqlen:seqlen + 1])
        out2, past2 = attn(x2, pos_emb2, past_key_value=past, use_cache=True, attention_mask=None)
        self.assertEqual(out2.shape, (bsz, 1, cfg.hidden_size))
        self.assertEqual(past2[0].shape[1], seqlen + 1)
        self.assertEqual(past2[1].shape[1], seqlen + 1)

    def test_feedforward_forward(self):
        cfg = _small_config(use_moe=False)
        ff = m.FeedForward(cfg).to(self.device)
        x = torch.randn(2, 3, cfg.hidden_size, device=self.device)
        y = ff(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_moe_gate_topk_and_aux_loss(self):
        cfg = _small_config(use_moe=True)
        gate = m.MoEGate(cfg).to(self.device)
        gate.train()

        bsz, seqlen = 2, 3
        x = torch.randn(bsz, seqlen, cfg.hidden_size, device=self.device)
        topk_idx, topk_w, aux = gate(x)

        # gate flattens tokens, so first dim is bsz*seqlen
        self.assertEqual(topk_idx.shape, (bsz * seqlen, cfg.num_experts_per_tok))
        self.assertEqual(topk_w.shape, (bsz * seqlen, cfg.num_experts_per_tok))
        self.assertTrue((topk_idx >= 0).all() and (topk_idx < cfg.n_routed_experts).all())
        self.assertTrue(torch.isfinite(topk_w).all())
        self.assertTrue(torch.isfinite(aux).all())

    def test_moe_feedforward_train_and_eval(self):
        cfg = _small_config(use_moe=True)
        moe = m.MOEFeedForward(cfg).to(self.device)

        x = torch.randn(2, 4, cfg.hidden_size, device=self.device)

        moe.train()
        y = moe(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(hasattr(moe, "aux_loss"))
        self.assertTrue(torch.isfinite(moe.aux_loss).all())

        moe.eval()
        with torch.no_grad():
            y2 = moe(x)
        self.assertEqual(y2.shape, x.shape)

    def test_minigpt_model_forward_and_cache(self):
        cfg = _small_config(use_moe=False)
        model = m.MiniGPTModel(cfg).to(self.device)
        model.eval()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 5), device=self.device)
        with torch.no_grad():
            hs, presents, aux = model(input_ids=input_ids, use_cache=True)
        self.assertEqual(hs.shape, (2, 5, cfg.hidden_size))
        self.assertEqual(len(presents), cfg.num_hidden_layers)
        self.assertTrue(torch.isfinite(hs).all())
        self.assertTrue(torch.isfinite(aux).all())

        # Next step with cache
        next_ids = torch.randint(0, cfg.vocab_size, (2, 1), device=self.device)
        with torch.no_grad():
            hs2, presents2, aux2 = model(input_ids=next_ids, past_key_values=presents, use_cache=True)
        self.assertEqual(hs2.shape, (2, 1, cfg.hidden_size))
        self.assertEqual(presents2[0][0].shape[1], 6)  # kv len should be 5+1
        self.assertTrue(torch.isfinite(aux2).all())

    def test_minigpt_model_moe_aux_loss(self):
        cfg = _small_config(use_moe=True)
        model = m.MiniGPTModel(cfg).to(self.device)
        model.train()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 5), device=self.device)
        hs, presents, aux = model(input_ids=input_ids, use_cache=False)
        self.assertEqual(hs.shape, (2, 5, cfg.hidden_size))
        self.assertTrue(torch.isfinite(aux).all())

    def test_causallm_forward_loss(self):
        cfg = _small_config(use_moe=False)
        lm = m.MiniGPTForCausalLM(cfg).to(self.device)
        lm.train()

        bsz, seqlen = 2, 6
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seqlen), device=self.device)
        labels = input_ids.clone()
        out = lm(input_ids=input_ids, labels=labels, use_cache=False, logits_to_keep=0)
        self.assertIsNotNone(out.loss)
        self.assertTrue(torch.isfinite(out.loss).all())
        self.assertEqual(out.logits.shape, (bsz, seqlen, cfg.vocab_size))
        self.assertTrue(hasattr(out, "aux_loss"))
        self.assertTrue(torch.isfinite(out.aux_loss).all())


def _run() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMiniGPTCoreModules)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_run())

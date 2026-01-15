import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model.model_minigpt import MiniGPTConfig, MiniGPTForCausalLM


def assert_raises(exc_type, fn, msg: str) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:  # pragma: no cover - unexpected exception path
        raise AssertionError(f"{msg}: raised {type(exc).__name__}") from exc
    raise AssertionError(f"{msg}: did not raise {exc_type.__name__}")


def test_moe_config_validation() -> None:
    def _build() -> None:
        cfg = MiniGPTConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            max_position_embeddings=64,
            use_moe=True,
            n_routed_experts=2,
            num_experts_per_tok=4,
        )
        MiniGPTForCausalLM(cfg)

    assert_raises(ValueError, _build, "MoE config validation")


def test_head_dim_even_validation() -> None:
    def _build() -> None:
        cfg = MiniGPTConfig(
            hidden_size=6,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            max_position_embeddings=32,
        )
        MiniGPTForCausalLM(cfg)

    assert_raises(ValueError, _build, "head_dim even validation")


def test_past_key_values_length_validation() -> None:
    cfg = MiniGPTConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=64,
    )
    model = MiniGPTForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))

    def _call() -> None:
        model.model(input_ids, past_key_values=[None])

    assert_raises(ValueError, _call, "past_key_values length validation")


def test_cache_consistency_flash_attn() -> None:
    torch.manual_seed(0)
    cfg = MiniGPTConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=64,
        flash_attn=True,
    )
    model = MiniGPTForCausalLM(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    full = model(input_ids, attention_mask=attention_mask).logits[:, -1, :]

    prefix = input_ids[:, :-1]
    prefix_mask = attention_mask[:, :-1]
    out1 = model(prefix, attention_mask=prefix_mask, use_cache=True)

    next_token = input_ids[:, -1:]
    next_mask = attention_mask[:, -1:]
    out2 = model(
        next_token,
        attention_mask=next_mask,
        past_key_values=out1.past_key_values,
        use_cache=True,
        logits_to_keep=1,
    )

    step = out2.logits[:, -1, :]
    assert torch.allclose(full, step, atol=1e-4, rtol=1e-4)


def _mask_invariance_check(model: MiniGPTForCausalLM, vocab_size: int) -> None:
    input_ids = torch.randint(0, vocab_size, (1, 5))
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    masked_pos = 2
    attention_mask[:, masked_pos] = 0

    input_ids_alt = input_ids.clone()
    input_ids_alt[:, masked_pos] = (input_ids_alt[:, masked_pos] + 1) % vocab_size

    out_a = model(input_ids, attention_mask=attention_mask).logits
    out_b = model(input_ids_alt, attention_mask=attention_mask).logits

    assert not torch.isnan(out_a).any()
    assert not torch.isnan(out_b).any()

    diff = (out_a[:, masked_pos + 1 :, :] - out_b[:, masked_pos + 1 :, :]).abs().max().item()
    assert diff < 1e-4


def test_attention_mask_invariance_flash() -> None:
    torch.manual_seed(1)
    cfg = MiniGPTConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=64,
        flash_attn=True,
    )
    model = MiniGPTForCausalLM(cfg)
    model.eval()
    _mask_invariance_check(model, cfg.vocab_size)


def test_attention_mask_invariance_noflash() -> None:
    torch.manual_seed(2)
    cfg = MiniGPTConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=64,
        flash_attn=False,
    )
    model = MiniGPTForCausalLM(cfg)
    model.eval()
    _mask_invariance_check(model, cfg.vocab_size)


def main() -> None:
    test_moe_config_validation()
    test_head_dim_even_validation()
    test_past_key_values_length_validation()
    test_cache_consistency_flash_attn()
    test_attention_mask_invariance_flash()
    test_attention_mask_invariance_noflash()
    print("Model feature checks passed.")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model.model_minigpt import MiniGPTConfig, MiniGPTForCausalLM


def main() -> None:
    config = MiniGPTConfig(
        vocab_size=6400,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        use_moe=True,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )
    model = MiniGPTForCausalLM(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params}")

    dummy_input = torch.randint(0, config.vocab_size, (2, 10))
    attention_mask = torch.ones_like(dummy_input, dtype=torch.long)
    output = model(dummy_input, attention_mask=attention_mask, use_cache=True)

    expected_shape = (2, 10, config.vocab_size)
    assert output.logits.shape == expected_shape
    print(f"Output shape: {output.logits.shape}")
    assert output.past_key_values is not None
    assert len(output.past_key_values) == config.num_hidden_layers
    k, v = output.past_key_values[0]
    expected_kv_shape = (2, config.num_key_value_heads, 10, config.hidden_size // config.num_attention_heads)
    assert k.shape == expected_kv_shape
    assert v.shape == expected_kv_shape
    assert output.aux_loss is not None
    print("Phase 2 check passed: forward output shape matches expectation.")


if __name__ == "__main__":
    main()

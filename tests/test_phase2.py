import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model.model_minigpt import MiniGPT, MiniGPTConfig


def main() -> None:
    config = MiniGPTConfig(
        vocab_size=6400,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
        moe_num_experts=4,
        moe_top_k=2,
    )
    model = MiniGPT(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params}")

    dummy_input = torch.randint(0, config.vocab_size, (2, 10))
    output = model(dummy_input)

    expected_shape = (2, 10, config.vocab_size)
    assert output.logits.shape == expected_shape
    print(f"Output shape: {output.logits.shape}")
    print("Phase 2 check passed: forward output shape matches expectation.")


if __name__ == "__main__":
    main()

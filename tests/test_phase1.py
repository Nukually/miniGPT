"""
Tokenizer evaluation / sanity checks.

This file contains the testing logic that was originally in train_tokenizer.py.
"""

from __future__ import annotations

from transformers import AutoTokenizer


def eval_tokenizer(tokenizer_dir: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自地球"},
    ]

    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    print("-" * 100)
    print(new_prompt)

    print("-" * 100)
    print("tokenizer词表长度：", len(tokenizer))

    model_inputs = tokenizer(new_prompt)
    print("encoder长度：", len(model_inputs["input_ids"]))

    response = tokenizer.decode(model_inputs["input_ids"], skip_special_tokens=False)
    print("decoder一致性：", response == new_prompt, "\n")

    print("-" * 100)
    print("流式解码（字节缓冲）测试：")

    input_ids = model_inputs["input_ids"]
    token_cache = []

    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)

        if current_decode and "\ufffd" not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in token_cache]
            print(
                f"Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}"
            )
            token_cache = []


def main() -> None:
    # Default directory (kept identical to the original script)
    tokenizer_dir = "../model/"
    eval_tokenizer(tokenizer_dir)


if __name__ == "__main__":
    main()

from transformers import PreTrainedTokenizerFast


def main() -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./model")

    text = "你是一个优秀的聊天机器人，总是给我正确的回应！"
    input_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    print(f"Original: {text}")
    print(f"Token IDs: {input_ids}")
    print(f"Decoded : {decoded_text}")

    normalized = decoded_text.replace(" ", "")
    assert normalized == text, "Decoded text does not match original text."
    print("Phase 1 check passed: tokenizer round-trip is consistent.")


if __name__ == "__main__":
    main()

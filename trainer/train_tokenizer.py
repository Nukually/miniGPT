#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def iter_jsonl_text(path: Path, text_key: str, limit: int | None):
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_key)
            if text:
                yield text


def build_tokenizer(
    texts,
    vocab_size: int,
    min_frequency: int,
    special_tokens: list[str],
):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/pretrain_hq.jsonl"),
        help="Path to the JSONL file containing a 'text' field.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("model"),
        help="Directory to save tokenizer.json and tokenizer_config.json.",
    )
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=DEFAULT_SPECIAL_TOKENS,
        help="Space-separated list of special tokens.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    output_tokenizer = args.output_dir / "tokenizer.json"
    if output_tokenizer.exists() and not args.overwrite:
        raise SystemExit(
            f"{output_tokenizer} already exists. Use --overwrite or --output_dir."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    texts = iter_jsonl_text(args.input, args.text_key, args.limit)
    tokenizer = build_tokenizer(
        texts=texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special_tokens,
    )

    eos_token = args.special_tokens[0] if args.special_tokens else None
    additional = args.special_tokens[1:] if len(args.special_tokens) > 1 else []
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=eos_token,
        additional_special_tokens=additional,
    )
    hf_tokenizer.save_pretrained(args.output_dir)

    print(f"Saved tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()

# Repository Guidelines

## Project Structure & Module Organization
Core model code lives in `model/` (notably `model_minigpt.py` plus tokenizer assets like `vocab.json` and `tokenizer.json`). Training scripts and utilities are in `trainer/`, while dataset loaders are defined in `dataset/lm_dataset.py`. Large training corpora live under `dataset/` (for example `pretrain_hq.jsonl`, `sft_mini_512.jsonl`, and `dpo.jsonl`) and are gitignored; add new data there and keep it out of commits. Tests are in `tests/` with phase1 (tokenizer sanity) and phase2 (core model) coverage. Training outputs are expected under `out/` (also gitignored).

## Build, Test, and Development Commands
- `cd trainer && python train_tokenizer.py` trains a ByteLevel BPE tokenizer from `../dataset/pretrain_hq.jsonl` and writes to `../model/`. Run from `trainer/` to use the default paths.
- `cd trainer && python train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl --save_dir ../out` runs pretraining; adjust flags like `--epochs`, `--hidden_size`, or `--use_moe` as needed.
- `cd tests && python test_phase1.py` prints tokenizer sanity checks against `../model/`.
- `python tests/test_phase2.py` runs CPU-friendly unit tests for the model stack.

## Coding Style & Naming Conventions
Use 4-space indentation and follow the existing formatting patterns. Classes are `CamelCase`, functions and variables are `snake_case`. Keep comments short and consistent with the current mixed English/Chinese usage. Avoid reflowing long lines unless you are already touching that block.

## Testing Guidelines
Tests use `unittest` with small configs to keep runtime low on CPU. New tests should be deterministic (seed RNGs) and focus on shapes, cache behavior, and finite outputs. Place new tests in `tests/` and follow the `test_phase*.py` naming pattern.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects, sometimes with tags (e.g., `[fix] attention mask and tests`, `[update] phase2 moe model`). Follow that style and keep messages focused. PRs should include a concise summary, the commands you ran (or “not run”), and notes on any new data or artifact paths. Do not commit large `.jsonl` datasets or generated weights.

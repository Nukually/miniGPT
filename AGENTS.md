# Repository Guidelines

## Project Structure & Module Organization
- `dataset/` holds JSONL training data (e.g., `dataset/pretrain_hq.jsonl`, `dataset/sft_mini_512.jsonl`, `dataset/dpo.jsonl`, `dataset/rlaif-mini.jsonl`).
- `model/` currently contains tokenizer assets (`model/tokenizer.json`, `model/tokenizer_config.json`).
- `scripts/` and `trainer/` are reserved for future CLI demos and training loops; see `LEARNING_GUIDE.md` for the planned layout and file names.
- `LEARNING_GUIDE.md` is the roadmap for building out code (model, dataset pipeline, training, inference, and tests).

## Build, Test, and Development Commands
- No build or test scripts are present in the repo today.
- Once you add training scripts following `LEARNING_GUIDE.md`, expect commands like:
  - `python trainer/train_pretrain.py --epochs 1 --batch_size 2 --save_dir ./out_test` for a short pretrain sanity run.
  - `python trainer/train_full_sft.py --epochs 1 --batch_size 2 --save_dir ./out_sft_test` for SFT validation.

## Development Environment
- Always perform development in the `minimind` conda environment (preconfigured for this repo).

## Coding Style & Naming Conventions
- Use Python 3 with 4-space indentation and type hints where helpful.
- Prefer clear, descriptive names that match the guide: `model/model_minigpt.py`, `dataset/lm_dataset.py`, `trainer/train_pretrain.py`.
- Keep small validation scripts named `test_phaseX.py` (e.g., `test_phase1.py`) as shown in `LEARNING_GUIDE.md`.

## Testing Guidelines
- No testing framework is configured yet.
- Follow the guide’s pattern for lightweight, script-based checks (e.g., `test_phase1.py`, `test_phase2.py`).
- If you introduce a formal test framework, place tests under `tests/` and document the command to run them.

## Commit & Pull Request Guidelines
- Commit messages in history follow a bracketed tag and short description, e.g., `[update] prompt prefill`.
- Keep commit subjects concise and action-oriented.
- PRs should describe the added milestone (e.g., “Phase 2 model architecture”), list new scripts/paths, and include sample commands or screenshots if a demo UI is added.
- After finishing code changes, push updates to GitHub.

## Data & Model Artifacts
- Treat `dataset/` and `model/` as large, versioned artifacts; avoid editing them unless you are regenerating the tokenizer or datasets intentionally.
- Keep new outputs (checkpoints, logs) under `out/` or another clearly named directory and exclude them from commits unless needed for reproducibility.

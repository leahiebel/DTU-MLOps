# Copilot / AI Agent Instructions

Purpose: Help an AI coding agent get productive quickly in this MLOps example repo (DTU course cookiecutter_project).

- **Big picture:** This is a small PyTorch-based ML project with three main concerns:
  - Data prep: convert and normalize raw .pt tensors into a single processed dataset ([cookiecutter_repo/src/cookiecutter_project/data.py](cookiecutter_repo/src/cookiecutter_project/data.py)).
  - Training & model: training loop produces a saved model at `models/model.pth` and training figures ([cookiecutter_repo/src/cookiecutter_project/train.py](cookiecutter_repo/src/cookiecutter_project/train.py), [cookiecutter_repo/src/cookiecutter_project/model.py](cookiecutter_repo/src/cookiecutter_project/model.py)).
  - Evaluation / API: evaluation loads the saved checkpoint and computes accuracy ([cookiecutter_repo/src/cookiecutter_project/evaluate.py](cookiecutter_repo/src/cookiecutter_project/evaluate.py)). An `api.py` module holds the service surface if an inference API is needed ([cookiecutter_repo/src/cookiecutter_project/api.py](cookiecutter_repo/src/cookiecutter_project/api.py)).

- **Data layout & conventions:**
  - Raw data: `data/raw/` contains multiple `train_images_*.pt` and `train_target_*.pt` shards and single `test_*.pt` files.
  - Processed data: `data/processed/` contains `train_images.pt`, `train_target.pt`, `test_images.pt`, `test_target.pt` produced by the `preprocess_data` CLI in `data.py`.
  - Models are saved to `models/model.pth`. Training figures go to `reports/figures/`.

- **CLI / run conventions:**
  - CLIs are implemented with `typer` and guarded by `if __name__ == "__main__":` so modules can be run directly.
  - Common examples observed in this repo's workflows and terminal history:
    - Preprocess raw data: `uv run src/cookiecutter_project/data.py data/raw data/processed`
    - Train: `uv run train`  (also runnable as `python -m cookiecutter_project.train` or `python src/cookiecutter_project/train.py`)
    - Evaluate: `python -m cookiecutter_project.evaluate models/model.pth`
  - `uv run` appears used in developer flows (task runner/alias); if missing, prefer `python -m <module>`.

- **Testing & CI:**
  - Tests live under `tests/` (unit tests: `tests/test_api.py`, `tests/test_data.py`, `tests/test_model.py`). Run locally with `pytest -q`.
  - A GitHub Actions workflow exists under `.github/workflows/` (see repo README for CI hints).

- **Patterns & expectations for edits:**
  - Keep data paths relative to the repository root (`data/processed`, `models/`). Tests expect these locations.
  - Preprocessing builds contiguous tensors by concatenating shards — work with tensors directly and preserve dtype/shape conventions (images -> `unsqueeze(1).float()`, targets -> `long()`). See [`data.py`](cookiecutter_repo/src/cookiecutter_project/data.py).
  - Training uses `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Respect device-aware code when adding new train/eval logic.
  - Model checkpointing is simple state_dict saving; load with `model.load_state_dict(torch.load(checkpoint_path))` as in [`evaluate.py`](cookiecutter_repo/src/cookiecutter_project/evaluate.py).

- **Files to inspect first for most tasks:**
  - Architecture & entrypoints: [cookiecutter_repo/src/cookiecutter_project/train.py](cookiecutter_repo/src/cookiecutter_project/train.py), [cookiecutter_repo/src/cookiecutter_project/data.py](cookiecutter_repo/src/cookiecutter_project/data.py)
  - Model definition: [cookiecutter_repo/src/cookiecutter_project/model.py](cookiecutter_repo/src/cookiecutter_project/model.py)
  - Evaluation and CLI patterns: [cookiecutter_repo/src/cookiecutter_project/evaluate.py](cookiecutter_repo/src/cookiecutter_project/evaluate.py)

- **When modifying or adding code:**
  - Run data preprocessing first (creates `data/processed/*.pt`) before training or tests that require processed data.
  - Preserve simple file-based artifacts (`models/model.pth`, `reports/figures/*`) used by tests and CI.
  - Prefer updating `src/cookiecutter_project/*.py` modules and keep top-level tasks.py or `uv` tasks in sync if you add new CLI commands.

If anything here is unclear or you want more detail about CI commands, `uv` usage, or test expectations, tell me which area to expand and I'll update this file. 

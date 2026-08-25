# Repository Guidelines

## Project Structure & Module Organization

Core library code lives in `src/openpi`, with major areas split into `models`, `models_pytorch`, `policies`, `serving`, `shared`, and `training`. Runnable utilities are in `scripts`, including training, validation, normalization statistics, policy serving, and robot inference helpers. Example integrations and data converters live under `examples`, documentation under `docs`, and the workspace client package under `packages/openpi-client`. Vendored dependencies and assets are kept in `third_party`; avoid editing them unless the change is intentionally vendor-specific.

## Build, Test, and Development Commands

- `GIT_LFS_SKIP_SMUDGE=1 uv sync`: install the pinned workspace dependencies from `uv.lock`.
- `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`: install the repository in editable mode after syncing.
- `uv run pytest`: run the default test suite from `src`, `scripts`, and `packages`.
- `uv run pytest src/openpi/models`: run a focused subset while iterating.
- `uv run ruff check .` and `uv run ruff format .`: lint and format Python code.
- `pre-commit install` then `pre-commit run --all-files`: run the same lockfile, lint, and format checks used before PRs.
- `uv run scripts/compute_norm_stats.py --config-name pi05_libero`: compute normalization stats for a training config.
- `uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite`: start training for a named config.

## Coding Style & Naming Conventions

Python targets 3.11 and is formatted with Ruff at a 120-character line length. Follow existing module patterns: lowercase module names, `snake_case` functions and variables, `PascalCase` classes, and `*_test.py` test files. Keep imports sorted by Ruff/isort; this repo prefers single-line imports. The linter excludes `third_party` and generated transformer replacement code.

## Testing Guidelines

Tests use `pytest`; `pyproject.toml` sets discovery to `src`, `scripts`, and `packages`. Add focused unit tests near the code they cover, following existing names such as `transforms_test.py` or `policy_test.py`. Mark hardware-heavy or manual checks with `@pytest.mark.manual`. For training and GPU changes, include at least a lightweight config or data-loader test when a full run is impractical.

## Commit & Pull Request Guidelines

Recent history uses concise task summaries, sometimes with prefixes like `feat:` or `add:`. Prefer short imperative commits that name the affected area, for example `feat: add LeRobot task metadata dropout`. PRs should include a clear title, a summary of behavior changes, reproduction or validation commands, and linked issues or discussion when relevant. For robot, inference, or UI-facing examples, include logs, screenshots, or sample outputs that make review possible.

## Security & Configuration Tips

Do not commit checkpoints, local datasets, credentials, W&B tokens, or machine-specific cache paths. Large model assets should stay in configured storage such as `gs://openpi-assets` or local ignored directories. Use `OPENPI_DATA_HOME` when overriding cache locations, and document non-default GPU, CUDA, or Docker assumptions in the PR.

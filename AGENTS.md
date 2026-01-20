# Repository Guidelines

This file is a contributor guide. For user-facing setup and full CLI usage, see `README.md`.

## Project Structure & Module Organization
- `src/futures_curve/`: core library code.
  - `stage0/`–`stage2/`: pipeline stages (metadata → ingestion → curve/spreads + roll metrics).
  - `cli.py`: CLI entrypoint (`python -m futures_curve.cli …` / `futures-curve …`).
- `config/`: YAML configs (e.g., `default.yaml`, `repro_20260119.yaml`).
- `tests/`: `pytest` suite + deterministic fixtures (keep fixtures small).
- Generated (not committed): `data_parquet/`, `metadata/`.

## Build, Test, and Development Commands
- `pip install -e .`: editable install for local development.
- `pytest -q`: run the full test suite.
- End-to-end pipeline + report commands are intentionally documented in `README.md` to avoid duplication.

## Coding Style & Naming Conventions
- Python, 4-space indentation, `snake_case` for functions/vars, `PascalCase` for classes.
- Prefer type hints and small, testable functions; avoid “magic” defaults—thread config via YAML/CLI.
- Do not introduce large generated artifacts into Git.

## Testing Guidelines
- Framework: `pytest` (see `pyproject.toml`).
- Naming: `tests/test_*.py`.
- For data pipeline changes (timezone, trade_date cutoff, curve ranking, spread math), add a regression test on a tiny deterministic slice.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and scoped to intent (e.g., “Fix …”, “Add …”).
- PRs should include: what changed, why, how to reproduce (reference the `README.md` command lines), and any key metric deltas.

## Configuration & Data Notes
- Raw data is local-only (not in Git). Use `config/local_paths.yaml` (gitignored) or update `paths.raw_data` in a config.
- Keep runs reproducible: prefer canonical output folders (`data_parquet/`, `metadata/`).

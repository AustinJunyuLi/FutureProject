# Repository Guidelines

This file is a contributor guide. For user-facing setup and full CLI usage, see `README.md`.

## Project Structure & Module Organization
- `src/futures_curve/`: core library code.
  - `stage0/`–`stage4/`: pipeline stages (metadata → ingestion → curve/spreads → analytics → backtests).
  - `cli.py`: CLI entrypoint (`python -m futures_curve.cli …` / `futures-curve …`).
  - `visualization/`: figure generation.
  - `reporting.py`: LaTeX/PDF report builders (tracked HG analysis PDF).
- `config/`: YAML configs (e.g., `default.yaml`, `repro_20260119.yaml`).
- `tests/`: `pytest` suite + deterministic fixtures (keep fixtures small).
- Generated (not committed): `data_parquet/`, `metadata/`, `research_outputs/` (exception below).

## Build, Test, and Development Commands
- `pip install -e .`: editable install for local development.
- `pytest -q`: run the full test suite.
- End-to-end pipeline + report commands are intentionally documented in `README.md` to avoid duplication.

## Coding Style & Naming Conventions
- Python, 4-space indentation, `snake_case` for functions/vars, `PascalCase` for classes.
- Prefer type hints and small, testable functions; avoid “magic” defaults—thread config via YAML/CLI.
- Do not introduce large generated artifacts into Git. The only tracked output is:
  - `research_outputs/report/HG_Analysis_Results_Report.pdf`
- Do not commit LaTeX build byproducts (`.aux`, `.log`, `.toc`, …). Keep the report source `.tex` local-only.

## Testing Guidelines
- Framework: `pytest` (see `pyproject.toml`).
- Naming: `tests/test_*.py`.
- For research logic changes (signals, execution, costs, calendars), add a regression test on a tiny deterministic slice (example pattern: `tests/test_eom_regression.py`).

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and scoped to intent (e.g., “Fix …”, “Add …”).
- PRs should include: what changed, why, how to reproduce (reference the `README.md` command lines), and any key metric deltas.
- If a change affects the report, regenerate and include the updated tracked PDF.

## Configuration & Data Notes
- Raw data is local-only (not in Git). Use `config/local_paths.yaml` (gitignored) or update `paths.raw_data` in a config.
- Keep runs reproducible: prefer canonical output folders (`data_parquet/`, `metadata/`, `research_outputs/`).

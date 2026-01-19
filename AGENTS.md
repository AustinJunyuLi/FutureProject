# Repository Guidelines

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
- `python -m futures_curve.cli run --config config/repro_20260119.yaml`: run stages end-to-end (HG by default).
- `python -m futures_curve.cli report --symbol HG --data data_parquet --out research_outputs --report-type analysis`: regenerate the HG analysis report (requires `pdflatex`).
- `pytest -q`: run the full test suite.

## Coding Style & Naming Conventions
- Python, 4-space indentation, `snake_case` for functions/vars, `PascalCase` for classes.
- Prefer type hints and small, testable functions; avoid “magic” defaults—thread config via YAML/CLI.
- Do not introduce large generated artifacts into Git. The only tracked output is:
  - `research_outputs/report/HG_Analysis_Results_Report.pdf`

## Testing Guidelines
- Framework: `pytest` (see `pyproject.toml`).
- Naming: `tests/test_*.py`.
- For research logic changes (signals, execution, costs, calendars), add a regression test on a tiny deterministic slice (example pattern: `tests/test_eom_regression.py`).

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and scoped to intent (e.g., “Fix …”, “Add …”).
- PRs should include: what changed, why, how to reproduce (`run`/`report` commands), and any key metric deltas.
- If a change affects the report, regenerate and include the updated tracked PDF; do not commit LaTeX build byproducts.

## Configuration & Data Notes
- Raw data is local-only (not in Git). Use `config/local_paths.yaml` (gitignored) or update `paths.raw_data` in a repro config.
- Keep runs reproducible: record the config used and ensure outputs write to canonical folders (`data_parquet/`, `metadata/`, `research_outputs/`).

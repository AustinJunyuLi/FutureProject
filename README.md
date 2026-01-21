# Futures Curve Pipeline

Deterministic expiry-ranked futures curve (F1..F12) construction, spread calculation (S1..S11), and roll-timing metrics for CME/COMEX-style contracts.

## What’s in this repo

- Full source code for the pipeline (Stages 0–2), tests, and CLI.
- Raw data and large generated outputs are **not** committed to Git (see Data Setup).

## Key conventions (CME/COMEX)

- Exchange timezone: `US/Central` (CT)
- Trade date boundary: **17:00 CT**
- Daily maintenance break: **16:00–16:59 CT** (treated as QC-only; often filtered from analysis)
- Curve labels: **F1..F12 by expiry timestamp only** (never by liquidity)
- Spreads: `S1 = F2 - F1`, …, `S11 = F12 - F11`
  - Stored in raw price units (e.g., USD/lb for HG)
  - Normalized versions `S*_pct` are also produced for descriptive analytics

## Installation

Editable install:

```bash
pip install -e .
```

### Required optional dependencies

Several pipeline stages rely on optional pandas engines and exchange calendars. For end-to-end execution you should install:

- `pyarrow` (or `fastparquet`) for Parquet read/write
- `pandas-market-calendars` for CME exchange business-day schedules (used for DTE + expiry ranking)

These are declared in `pyproject.toml`.

If `pandas-market-calendars` is missing, the code falls back to a deterministic Mon–Fri calendar (holidays ignored). This is adequate for unit tests and local development, but is **not** appropriate for reproducing research numbers.

## Data setup (no Git / no upload)

We do **not** commit raw data (20+ GB) to GitHub. Each collaborator should place the data locally and point the config to it.

### Expected directory structure

The Stage 1 file scanner expects commodity folders under a single root:

```
/path/to/organized_data/
  copper/   # HG
  gold/     # GC
  silver/   # SI
  crude_oil/# CL
  natural_gas/ # NG
  corn/     # ZC
  soybeans/ # ZS
  wheat/    # ZW
```

### Expected file pattern

Each file is a 1‑minute OHLCV TXT/CSV with **no header** and these columns:

```
(timestamp, open, high, low, close, volume)
```

Filename pattern (per `FileScanner`):

```
{SYMBOL}_{MONTH_CODE}{YY}_1min.txt
# Example: HG_G22_1min.txt
```

Timestamps are **naive** strings in `YYYY-MM-DD HH:MM:SS`. The pipeline infers the raw timezone (commonly `US/Eastern` or `UTC`) by minimizing activity during the CME maintenance hour when converted to CT.

### Pointing the pipeline at your data

Option A — create a local config file (recommended):

Copy the default config to a gitignored local config file:

```bash
cp config/default.yaml config/local_paths.yaml
```

Edit `paths.raw_data` in `config/local_paths.yaml` to point at your local `organized_data` root, then run:

```bash
python -m futures_curve.cli run --config config/local_paths.yaml
```

Option B — run Stage 1 with an explicit input directory:

```bash
python -m futures_curve.cli stage1 --symbol HG --input /path/to/organized_data --output data_parquet
```

## Usage (end‑to‑end)

Run the full pipeline (Stages 0–2) for HG using the default config:

```bash
python -m futures_curve.cli run --config config/default.yaml
```

If you created `config/local_paths.yaml`, run with that instead:

```bash
python -m futures_curve.cli run --config config/local_paths.yaml
```

Or run stages explicitly:

```bash
python -m futures_curve.cli stage0 --symbol HG --output metadata
python -m futures_curve.cli stage1 --symbol HG --input /path/to/organized_data --output data_parquet
python -m futures_curve.cli stage2 --symbol HG --data data_parquet
```

The CLI entrypoint also works once installed:

```bash
futures-curve --help
```

## Configuration

Configuration is YAML-based. See `config/default.yaml`.

Key fields:

- `paths.raw_data`: root directory of raw 1-minute files
- `paths.output_parquet`: stage outputs (parquet)
- `paths.metadata`: expiry tables, contract specs
- `commodities`: list of symbols to process
- `ingestion.*`: chunk size, timezone inference settings, trade date cutoff
- `curve.*`: max contracts, min DTE threshold
- `roll.*`: roll thresholds, smoothing, persistence

## Pipeline stage overview

- **Stage 0**: expiry schedule + contract specs + trading calendar
- **Stage 1**: parse raw 1-minute data; aggregate to hourly buckets and daily bars
- **Stage 2**: deterministic curve panel (F1..F12), roll share/events, spreads

## Technical implementation details (from the technical report)

### Data source and format

- **Source:** CME Group via data vendor (1‑minute OHLCV bars)
- **Coverage:** All listed contract months (2008–2024 in the research sample)
- **Format:** CSV/TXT, no header, columns: `timestamp, open, high, low, close, volume`

### Time handling and trade date

- All analytics are normalized to **US/Central**.
- The **trade date** boundary is **17:00 CT** (CME Globex reopen).
- The **maintenance hour** is **16:00–16:59 CT**, stored as bucket 0 for QC.
- Raw timestamps are naive and can be vendor‑specific; the pipeline infers raw timezone by minimizing “maintenance hour leakage.”

### Bucket schema

Buckets are defined in exchange time (CT):

- 0: 16:00–16:59 (maintenance, QC only)
- 1–7: 09:00–15:59 (US session hours)
- 8: 17:00–20:59 (post‑reopen)
- 9: 21:00–02:59 (overnight)
- 10: 03:00–08:59 (pre‑US)

### Expiry and deterministic curve construction

- Expiries are calculated using CME business days (via `pandas-market-calendars`).
- **Eligibility:** contract is included only if `expiry_ts_utc > as_of_ts_utc`.
- **Ranking:** F1..F12 are assigned **strictly by expiry timestamp**, never by liquidity.
- Missing prints at a timestamp yield `NaN` price and `0` volume (no forward fill).

### Spreads and normalization

- Spreads are defined as `S1 = F2 − F1`, `S2 = F3 − F2`, …, `S11 = F12 − F11`.
- Normalized spreads (`S*_pct`) are computed as `S* / F*`.
- Rolling z‑scores are computed using **causal** windows (backward‑looking only).

### Roll detection (liquidity migration proxy)

- Volume share: `s(t) = V2 / (V1 + V2)`.
- Thresholds (defaults): start 0.25, peak 0.50, end 0.75.
- **Persistence:** requires consecutive confirmations (causal event timing).
- **Smoothing:** causal rolling mean over last K observations.
- Roll events are stored at both bucket and daily frequency; daily uses the last US‑session bucket for that trade date.

## Validation / sanity checks

- `pytest -q` should pass after install.
- Stage 1 QC report flags OHLC consistency and maintenance-hour leakage.
- Bucket 0 activity should be near zero in clean data.

## Outputs

- `metadata/`: calendars and expiry tables (generated; not committed)

## Analysis (research / local)

The pipeline (Stages 0–2) produces deterministic curve + spreads panels. The following research scripts build on those
Stage2 outputs and write results to `output/` (gitignored).

### Strategy scans (HG, S1–S4)

Inputs required:

- `data_parquet/spreads/HG/spreads_panel.parquet`

Carry / roll-down scan (Strategy family A):

```bash
python scripts/run_spread_carry.py --symbol HG --out output/analysis --top-n 50 --max-dd 0.15
```

DTE-conditioned mean reversion scan (Strategy family B):

```bash
python scripts/run_spread_mean_reversion.py --symbol HG --out output/analysis --top-n 50 --max-dd 0.15
```

Walk-forward OOS test (A + B, yearly re-selection):

```bash
python scripts/run_walk_forward_ab.py --symbol HG --out output/analysis/wf --roll-filter exclude_roll --max-dd 0.15
```

### LaTeX report (HG spread strategy search)

This report pulls together the walk-forward results into a small set of standardized figures/tables.

1) Generate report assets (figures + LaTeX table snippets) under `output/`:

```bash
python scripts/build_hg_spread_strategy_search_report.py
```

2) Compile the PDF (build artifacts go to `output/`, PDF is copied to `reports/`):

```bash
cd reports
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../output/.latex_build/hg_spread_strategy_search hg_spread_strategy_search.tex
cp ../output/.latex_build/hg_spread_strategy_search/hg_spread_strategy_search.pdf hg_spread_strategy_search.pdf
latexmk -c -outdir=../output/.latex_build/hg_spread_strategy_search hg_spread_strategy_search.tex
```

Notes:

- Signals use a US-session VWAP proxy (volume-weighted bucket closes across buckets 1–7).
- Execution is modeled at the next trade_date’s earliest available US-session bucket (bucket 1 preferred; falls back to 2–7 if needed).
- Costs use `HG` tick/value from `src/futures_curve/stage0/contract_specs.py` and assume **1 tick per leg per side**
  for spread trading (2 legs).
- `data_parquet/`: bucket data, curve panel, spreads, roll events (generated; not committed)

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

### Analysis framework internals (implementation details)

All analysis code lives under `src/futures_curve/analysis/` and is designed to be:

- **Causal** (signals observed on trade date `t`, executed on `t+1`)
- **DTE-aware in business days** (uses Stage2 `*_dte_bdays`)
- **Net of costs** (tick-based costs applied on each position change)
- **Risk-normalized** (vol-targeted position sizing)

Key modules:

- Data extraction: `src/futures_curve/analysis/data.py`
- Backtest engine: `src/futures_curve/analysis/backtest.py`
- Walk-forward selection: `src/futures_curve/analysis/walk_forward.py`
- Roll/liquidity filters: `src/futures_curve/analysis/roll.py`
- Strategy building blocks:
  - Carry proxy: `src/futures_curve/analysis/strategies/carry.py`
  - DTE-conditioned MR: `src/futures_curve/analysis/strategies/mean_reversion.py`

#### Daily series construction (from Stage2 spreads panel)

The Stage2 spread panel (`data_parquet/spreads/HG/spreads_panel.parquet`) is bucket-level with:

- `trade_date`, `bucket`
- per-leg prices and volumes: `F1_price`, `F1_volume`, …, `F12_price`, `F12_volume`
- per-leg business DTE: `F1_dte_bdays`, …, `F12_dte_bdays`

For spread `Sk = F{k+1} - F{k}`, `build_daily_spread_series()` produces one row per `trade_date`:

- Near/far contract ids: `near_contract`, `far_contract`
- Near/far DTE (business days): `near_dte_bdays`, `far_dte_bdays`
- **Signal prices (observed on trade date `t`)**
  - `fnear_signal`, `ffar_signal`: US-session VWAP proxy from buckets 1–7 (volume-weighted bucket closes)
  - `s_signal = ffar_signal - fnear_signal`
  - `s_signal_pct = s_signal / fnear_signal`
- **Execution prices (used on trade date `t+1`)**
  - Select the earliest available US bucket in `bucket ∈ [1..7]` where *both legs print* (bucket 1 preferred; else 2–7)
  - `fnear_exec`, `ffar_exec` are bucket closes at that selected bucket
  - `s_exec = ffar_exec - fnear_exec`
  - `exec_bucket` records which bucket was used

This execution “fallback” rule exists because bucket 1 can be thin/missing on some days; using the earliest available
US bucket materially improves sample coverage without changing the causal timing convention.

#### Backtest conventions (risk + costs)

Backtests (`run_backtest_single_series()`):

- **Signal timing:** signal observed on `t`, executed on `t+1` (implemented via `signal.shift(1)`).
- **PnL convention:** mark-to-market on execution prices:
  - One-contract daily PnL: `pnl_1[t] = (s_exec[t] - s_exec[t-1]) * contract_size`.
- **Vol targeting:** compute EWMA volatility of `pnl_1` and size to a daily dollar-vol target:
  - `target_dollar_vol = capital * vol_target_annual / sqrt(252)`
  - `position_target[t] = signal_for_exec[t] * target_dollar_vol / vol_1[t]`
- **Costs:** tick-based costs applied on each position change:
  - Per-execution cost: `abs(Δpos) * legs * ticks_per_leg_per_side * dollars_per_tick`
  - Spread trading uses `legs=2`.
  - HG tick size is `$0.0005/lb` with 25,000 lb contract size → `$12.50/tick`.
  - Net PnL: `pnl_net = pnl_gross - costs`
- **Metrics:** annualized Sharpe from daily returns, max drawdown from the equity curve, turnover from |Δposition|.

#### Strategy family A: carry / roll-down (adjacent spreads)

Carry proxy (annualized):

- `spacing_bdays = far_dte_bdays - near_dte_bdays`
- `spacing_years = spacing_bdays / 252`
- `carry = s_signal_pct / spacing_years`

Signals evaluated in the walk-forward search:

- **sign:** `sig = -sign(carry) * 1[|carry| >= threshold]`
- **clipped:** `sig = clip(-carry/scale, -1, +1) * 1[|carry| >= threshold]`

We also allow `direction ∈ {+1, -1}` (flip long/short) as part of the search.

#### Strategy family B: DTE-conditioned mean reversion (robust z-score)

We compute a causal, DTE-binned robust z-score on `s_signal_pct`:

1) Bin by near-leg DTE: `bin_id = floor(near_dte_bdays / dte_bin_size)`
2) Within each bin, compute rolling **median** and rolling **MAD** over `lookback_days`
3) `z = (s_signal_pct - median) / MAD`

Positions are discrete with hysteresis (enter at |z| ≥ entry_z, exit when reverted inside exit_z), with optional
`max_hold_days` to limit holding times.

#### Regime filters (used during walk-forward selection)

We apply simple, low-degree-of-freedom regime filters as masks on the signal day:

- **Roll/liquidity filter** from `data_parquet/roll_events/HG/roll_shares.parquet` (daily frequency):
  - `volume_share_smooth = V2 / (V1 + V2)` (smoothed, causal)
  - `exclude_roll`: tradable when share ≤ start OR share ≥ end
  - `pre_roll_only`: tradable when share ≤ start
  - `post_roll_only`: tradable when share ≥ end
- **Contango/backwardation filter** from `s_signal_pct`:
  - `contango_only`: `s_signal_pct > 0`
  - `backwardation_only`: `s_signal_pct < 0`
- **Vol regime filter**:
  - Vol series is the EWMA vol of 1-contract daily PnL (`pnl_1`)
  - Threshold = median vol over the training window
  - `high_vol_only`: vol ≥ threshold
  - `low_vol_only`: vol < threshold

To avoid a full cross-product explosion, the CLI uses a small set of (contango_mode, vol_mode) combinations.

#### Walk-forward evaluation (yearly expanding window)

Walk-forward (`scripts/run_walk_forward_ab.py`, `src/futures_curve/analysis/walk_forward.py`):

- For each test year `Y`:
  - Train = all dates ≤ Dec 31 of `Y-1`
  - Test = dates in calendar year `Y`
  - Select parameters on the training window by maximizing **train Sharpe** subject to gates.
- Gates (defaults; tunable via CLI/config):
  - training exposure: `train_nonzero_days >= min_train_nonzero_days`
  - test exposure: `test_nonzero_days >= min_test_nonzero_days`
  - holding-period proxy: turnover implies average holding in `[1, 20]` business days
  - maxDD constraint: `max_drawdown <= max_dd`
  - net mean daily return must be positive on train
- We then **stitch** the OOS PnL across years that produced eligible folds.
  - Note: if a year has no eligible fold, it is omitted from the stitched series (reported explicitly as `oos_years`).

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

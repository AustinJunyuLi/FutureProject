"""Lightweight LaTeX report builder for the futures-curve pipeline.

This generates two self-contained PDF reports:
- Technical Implementation Report: Explains the codebase conceptually
- Analysis Results Report: Presents findings with professional visualizations
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional, Literal

# Sandbox-friendly defaults
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

from .utils.timezone import BUCKETS


# Glossary of terms for the technical report
GLOSSARY_TERMS = {
    "F1": "Front-month contract; the contract with the nearest expiry date.",
    "F2": "Second-month contract; the contract with the second-nearest expiry.",
    "F1-F12": "Contract labels ranked strictly by expiry timestamp (F1 = nearest, F12 = 12th nearest).",
    "S1": "Front spread; calculated as S1 = F2 - F1 (second month minus front month).",
    "S1_raw": "Spread in price units (e.g., dollars per pound for copper).",
    "S1_pct": "Normalized spread as percentage: (F2 - F1) / F1.",
    "DTE": "Days to Expiry; business days remaining until contract expiration.",
    "Contango": "Market state where S1 > 0 (deferred contracts trade at premium to front).",
    "Backwardation": "Market state where S1 < 0 (front contracts trade at premium to deferred).",
    "Roll": "The transition of trading activity from F1 to F2 as F1 approaches expiry.",
    "Volume Share": "Ratio of F2 volume to total F1+F2 volume; used to detect roll timing.",
    "Bucket": "Hourly time period for intraday aggregation (e.g., 09:00-09:59 CT).",
    "CT": "Central Time (US/Central); the CME exchange timezone.",
    "Trade Date": "CME convention where each trade date starts at 17:00 CT the prior calendar day.",
    "Sharpe Ratio": "Risk-adjusted return metric: annualized mean return / annualized std dev.",
    "Drawdown": "Peak-to-trough decline in equity, expressed as percentage.",
    "Win Rate": "Percentage of trades that are profitable.",
    "Profit Factor": "Gross profits divided by gross losses (values > 1 indicate profitability).",
}

# CLI commands reference
CLI_COMMANDS = {
    "run": {
        "description": "Execute the full pipeline (stages 0-4)",
        "options": [
            ("--config", "Path to YAML configuration file"),
            ("--commodities", "Comma-separated list of symbols to process"),
            ("--stages", "Specific stages to run (default: all)"),
        ],
        "example": "python -m futures_curve.cli run --config config/default.yaml --commodities HG",
    },
    "report": {
        "description": "Generate PDF reports from pipeline outputs",
        "options": [
            ("--symbol", "Commodity symbol (e.g., HG)"),
            ("--data", "Data directory (parquet outputs)"),
            ("--out", "Research outputs directory"),
            ("--report-type", "Report type: technical, analysis, or both"),
        ],
        "example": "python -m futures_curve.cli report --symbol HG --report-type both",
    },
    "status": {
        "description": "Check pipeline status and data availability",
        "options": [
            ("--symbol", "Commodity symbol to check"),
        ],
        "example": "python -m futures_curve.cli status --symbol HG",
    },
}

# Default configuration parameters
DEFAULT_CONFIG = {
    "Pipeline Settings": [
        ("data_source", "/path/to/organized_data", "Root directory for raw tick data"),
        ("output_dir", "data_parquet", "Directory for processed parquet files"),
        ("research_dir", "research_outputs", "Directory for figures and reports"),
    ],
    "Backtesting Parameters": [
        ("slippage_ticks", "1", "Slippage per fill in tick units"),
        ("commission_per_contract", "2.50", "Commission per contract per side (USD)"),
        ("tick_value", "12.50", "Dollar value per tick (HG)"),
    ],
    "Pre-Expiry Strategy": [
        ("entry_dte", "5", "Enter when F1_dte_bdays <= entry_dte and > exit_dte (expiry-anchored)"),
        ("exit_dte", "1", "Exit when F1_dte_bdays <= exit_dte (expiry-anchored)"),
        ("direction", "long", "Trade direction (long/short spread)"),
        ("sweep", "entry_dte=2..10; exit_dte=0..3", "Grid-search around the last few business days before expiry"),
    ],
}


@dataclass(frozen=True)
class ReportPaths:
    data_dir: Path
    research_dir: Path
    tables_dir: Path
    figures_dir: Path
    report_dir: Path


def _read_parquet_optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _df_to_latex(
    df: pd.DataFrame,
    *,
    floatfmt: str = ".4f",
    index: bool = False,
    resize: bool = True,
    column_format: str | None = None,
) -> str:
    if df is None or df.empty:
        return "\\emph{(No data)}"
    table = df.to_latex(index=index, escape=True,
                        float_format=lambda x: format(x, floatfmt), bold_rows=False, column_format=column_format)
    if resize and len(df.columns) > 5:
        # Wrap wide tables in resizebox to fit page width
        return f"\\resizebox{{\\textwidth}}{{!}}{{{table}}}"
    return table


def build_report_paths(data_dir: str | Path, research_dir: str | Path) -> ReportPaths:
    data_dir = Path(data_dir)
    research_dir = Path(research_dir)
    return ReportPaths(
        data_dir=data_dir,
        research_dir=research_dir,
        tables_dir=research_dir / "tables",
        figures_dir=research_dir / "figures",
        report_dir=research_dir / "report",
    )


def _bucket_table() -> pd.DataFrame:
    rows = []
    for bid in sorted(BUCKETS.keys()):
        b = BUCKETS[bid]
        rows.append({
            "Bucket": bid,
            "Start (CT)": b.start.strftime("%H:%M"),
            "End (CT)": b.end.strftime("%H:%M"),
            "Description": b.description,
        })
    return pd.DataFrame(rows)


def _glossary_to_latex() -> str:
    """Convert glossary terms to LaTeX description list."""
    lines = ["\\begin{description}[style=nextline,leftmargin=2cm]"]
    for term, definition in sorted(GLOSSARY_TERMS.items()):
        # Escape special LaTeX characters
        term_esc = term.replace("_", "\\_").replace("%", "\\%")
        def_esc = definition.replace("_", "\\_").replace("%", "\\%")
        lines.append(f"  \\item[{term_esc}] {def_esc}")
    lines.append("\\end{description}")
    return "\n".join(lines)


def _cli_reference_to_latex() -> str:
    """Convert CLI commands to LaTeX format."""
    lines = []
    for cmd, info in CLI_COMMANDS.items():
        lines.append(f"\\subsection*{{\\texttt{{futures\\_curve {cmd}}}}}")
        lines.append(f"{info['description']}")
        lines.append("")
        lines.append("\\textbf{Options:}")
        lines.append("\\begin{itemize}[nosep]")
        for opt, desc in info['options']:
            lines.append(f"  \\item \\texttt{{{opt}}}: {desc}")
        lines.append("\\end{itemize}")
        lines.append("")
        lines.append("\\textbf{Example:}")
        lines.append(f"\\begin{{verbatim}}\n{info['example']}\n\\end{{verbatim}}")
        lines.append("")
    return "\n".join(lines)


def _config_table_to_latex() -> str:
    """Convert default config to LaTeX tables."""
    lines = []
    for section, params in DEFAULT_CONFIG.items():
        lines.append(f"\\textbf{{{section}:}}")
        lines.append("\\begin{center}")
        lines.append("\\begin{tabular}{|l|l|p{6cm}|}")
        lines.append("\\hline")
        lines.append("\\textbf{Parameter} & \\textbf{Default} & \\textbf{Description} \\\\")
        lines.append("\\hline")
        for param, default, desc in params:
            # Escape all special LaTeX characters
            param_esc = param.replace("_", "\\_").replace("/", "/\\allowbreak ")
            default_esc = default.replace("_", "\\_").replace("/", "/\\allowbreak ")
            desc_esc = desc.replace("_", "\\_")
            lines.append(f"{param_esc} & \\texttt{{{default_esc}}} & {desc_esc} \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{center}")
        lines.append("")
    return "\n".join(lines)


def _get_sample_trades(trades_path: Path, n: int = 10) -> Optional[pd.DataFrame]:
    """Load first N trades from trade log."""
    log_path = trades_path / "trade_log.parquet"
    if log_path.exists():
        df = pd.read_parquet(log_path)
    else:
        # Fall back to any strategy trades file (e.g., pre_expiry_trades.parquet).
        trade_files = sorted(trades_path.glob("*_trades.parquet"))
        if not trade_files:
            return None
        df = pd.read_parquet(trade_files[0])
    if df.empty:
        return None
    # Select key columns for display
    display_cols = ["entry_date", "exit_date", "direction", "entry_price", "exit_price", "pnl_net", "pnl"]
    available_cols = [c for c in display_cols if c in df.columns]
    if not available_cols:
        return df.head(n)
    return df[available_cols].head(n)


def build_technical_report(
    symbol: str,
    data_dir: str | Path = "data_parquet",
    research_dir: str | Path = "research_outputs",
    compile_pdf: bool = True,
) -> Path:
    """Build the Technical Implementation Report.

    Audience: Non-technical academic manager
    Purpose: Explain what the codebase does conceptually
    """
    paths = build_report_paths(data_dir, research_dir)
    paths.report_dir.mkdir(parents=True, exist_ok=True)

    bucket_tbl = _bucket_table()

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\usepackage{{float}}
\usepackage{{enumitem}}
\usepackage{{xcolor}}
\usepackage{{tocloft}}
\usepackage{{longtable}}

\definecolor{{primary}}{{HTML}}{{2C3E50}}
\definecolor{{positive}}{{HTML}}{{27AE60}}
\definecolor{{negative}}{{HTML}}{{C0392B}}

\hypersetup{{
    colorlinks=true,
    linkcolor=primary,
    urlcolor=primary,
}}

\title{{\textbf{{{symbol} Futures Curve Pipeline}}\\[0.5em]\Large Technical Implementation Report}}
\author{{futures\_curve Pipeline Documentation}}
\date{{\today}}

\begin{{document}}
\maketitle
\tableofcontents
\newpage

%==============================================================================
\section{{Executive Summary}}
%==============================================================================

This document provides a technical overview of the \texttt{{futures\_curve}} pipeline, designed
to analyze commodity futures curve dynamics and backtest spread trading strategies. The pipeline
processes raw tick data through multiple stages, ultimately producing analysis outputs and
backtest results.

\textbf{{Key Capabilities:}}
\begin{{itemize}}
    \item Ingest and normalize raw futures tick data from vendor CSV files
    \item Construct continuous futures curves (F1 through F12) with deterministic contract ranking
    \item Calculate and analyze calendar spreads (S1 = F2 - F1)
    \item Detect contract roll events using volume share thresholds
    \item Run backtests on expiry-anchored (\texttt{{pre\_expiry}}) and liquidity-trigger spread strategies
    \item Model realistic transaction costs (slippage + commissions)
    \item Generate publication-quality visualizations and PDF reports
\end{{itemize}}

\textbf{{Pipeline Outputs:}}
\begin{{itemize}}
    \item Hourly bucket aggregates (OHLCV) for each contract
    \item Continuous curve panels at bucket and daily frequencies
    \item Spread time series with normalization
    \item Roll event detection with timing metrics
    \item Trade-by-trade backtest logs with P\&L attribution
    \item Performance analytics (Sharpe, drawdown, win rate, profit factor)
\end{{itemize}}

%==============================================================================
\section{{System Architecture}}
%==============================================================================

\subsection{{Data Flow Overview}}

The pipeline processes data through five sequential stages, each building on the outputs
of previous stages:

\begin{{center}}
\begin{{tabular}}{{|c|l|l|p{{6cm}}|}}
\hline
\textbf{{Stage}} & \textbf{{Name}} & \textbf{{Input}} & \textbf{{Output}} \\
\hline
0 & Metadata & CME specs & Expiry calendar, trading calendar \\
1 & Ingestion & Raw CSV ticks & Hourly bucket parquet files \\
2 & Curve & Bucket parquet & Curve panels, spreads, roll events \\
3 & Analysis & Curve panels & Seasonal/DTE summaries, event studies \\
4 & Backtesting & Spreads + rules & Trade logs, performance metrics \\
\hline
\end{{tabular}}
\end{{center}}

\subsection{{Stage-by-Stage Outputs}}

\textbf{{Stage 0: Metadata Generation}}

Outputs stored in \texttt{{data\_parquet/metadata/<SYMBOL>/}}:
\begin{{itemize}}
    \item \texttt{{expiry\_calendar.parquet}} --- Contract codes with expiry timestamps
    \item \texttt{{trading\_calendar.parquet}} --- Valid trading days with holiday flags
    \item \texttt{{contract\_specs.parquet}} --- Tick size, multiplier, trading hours
\end{{itemize}}

\textbf{{Stage 1: Data Ingestion}}

Outputs stored in \texttt{{data\_parquet/buckets/<SYMBOL>/}}:
\begin{{itemize}}
    \item One parquet file per contract (e.g., \texttt{{HGH24.parquet}})
    \item Schema: \texttt{{trade\_date, bucket\_id, open, high, low, close, volume, vwap}}
    \item 10 hourly buckets per trade date (see Section 3.3)
\end{{itemize}}

\textbf{{Stage 2: Curve Construction}}

Outputs stored in \texttt{{data\_parquet/curve/<SYMBOL>/}} and \texttt{{spreads/<SYMBOL>/}}:
\begin{{itemize}}
    \item \texttt{{curve\_panel.parquet}} --- Columns: F1\_price, F2\_price, ..., F12\_price
    \item \texttt{{spread\_panel.parquet}} --- Columns: S1\_raw, S1\_pct, DTE
    \item \texttt{{roll\_events.parquet}} --- Roll start/peak/end dates per contract
\end{{itemize}}

\textbf{{Stage 3: Analysis}}

Outputs stored in \texttt{{research\_outputs/tables/<SYMBOL>\_*.parquet}}:
\begin{{itemize}}
    \item Seasonal summary by month
    \item DTE profile by 5-day bins
    \item Roll event study statistics
    \item Data quality diagnostics
\end{{itemize}}

\textbf{{Stage 4: Backtesting}}

Outputs stored in \texttt{{data\_parquet/trades/<SYMBOL>/}}:
\begin{{itemize}}
    \item \texttt{{trade\_log.parquet}} --- Entry/exit timestamps, prices, P\&L
    \item \texttt{{equity\_curve.parquet}} --- Cumulative P\&L time series
    \item \texttt{{summary.parquet}} --- Aggregate performance metrics
\end{{itemize}}

%==============================================================================
\section{{Data Inputs and Metadata}}
%==============================================================================

\subsection{{Input File Format}}

The pipeline expects vendor tick data in CSV format with the following structure:

\begin{{verbatim}}
DateTime,Open,High,Low,Close,Volume
2024-01-02 17:00,3.8575,3.8590,3.8570,3.8585,125
2024-01-02 17:01,3.8585,3.8600,3.8580,3.8595,89
...
\end{{verbatim}}

Files are named by contract code and frequency (e.g., \texttt{{HGH24\_1min.txt}} for
March 2024 copper at 1-minute frequency).

\subsection{{Expiry Calculation Rules}}

Contract expiration dates follow CME specifications:
\begin{{itemize}}
    \item \textbf{{Copper (HG):}} Third-to-last business day of the contract month
    \item \textbf{{Gold (GC):}} Third-to-last business day of the contract month
    \item \textbf{{Crude Oil (CL):}} Three business days prior to the 25th calendar day
\end{{itemize}}

The pipeline computes expiry dates algorithmically and stores them in the metadata stage.

\subsection{{Contract Specifications}}

\begin{{center}}
\begin{{tabular}}{{|l|c|c|c|c|}}
\hline
\textbf{{Symbol}} & \textbf{{Product}} & \textbf{{Tick Size}} & \textbf{{Tick Value}} & \textbf{{Multiplier}} \\
\hline
HG & Copper & 0.0005 & \$12.50 & 25,000 lbs \\
GC & Gold & 0.10 & \$10.00 & 100 oz \\
CL & Crude Oil & 0.01 & \$10.00 & 1,000 bbl \\
\hline
\end{{tabular}}
\end{{center}}

\subsection{{Time Conventions}}

\begin{{itemize}}
    \item \textbf{{Exchange Timezone:}} US/Central (CT) for CME products
    \item \textbf{{Trade Date Boundary:}} 17:00 CT marks the start of each trade date
    \item \textbf{{Example:}} Activity at 18:00 CT on Monday belongs to Tuesday's trade date
\end{{itemize}}

\subsection{{Bucket Schema}}

Trading activity is aggregated into 10 hourly buckets plus a maintenance period:

{_df_to_latex(bucket_tbl)}

%==============================================================================
\section{{Pipeline Stages}}
%==============================================================================

\subsection{{Stage 0: Metadata}}

\textbf{{Purpose:}} Build reference tables for contract expiries and trading calendars.

\textbf{{Process:}}
\begin{{enumerate}}
    \item Parse CME product specifications
    \item Generate expiry dates for all contract months in date range
    \item Build trading calendar with US market holidays removed
    \item Store as parquet files for downstream stages
\end{{enumerate}}

\textbf{{Output Schema (expiry\_calendar):}}
\begin{{center}}
\begin{{tabular}}{{|l|l|l|}}
\hline
\textbf{{Column}} & \textbf{{Type}} & \textbf{{Description}} \\
\hline
contract\_code & string & Contract identifier (e.g., HGH24) \\
expiry\_date & datetime & Expiration timestamp \\
contract\_month & int & Delivery month (1-12) \\
contract\_year & int & Delivery year \\
\hline
\end{{tabular}}
\end{{center}}

\subsection{{Stage 1: Ingestion}}

\textbf{{Purpose:}} Parse raw tick data and create hourly bucket aggregates.

\textbf{{Process:}}
\begin{{enumerate}}
    \item Read vendor CSV files for each contract
    \item Convert timestamps to Central Time
    \item Assign trade date using 17:00 CT boundary
    \item Assign bucket ID based on hour
    \item Aggregate to OHLCV per bucket
    \item Write parquet files per contract
\end{{enumerate}}

\textbf{{Output Schema (bucket parquet):}}
\begin{{center}}
\begin{{tabular}}{{|l|l|l|}}
\hline
\textbf{{Column}} & \textbf{{Type}} & \textbf{{Description}} \\
\hline
trade\_date & date & CME trade date \\
bucket\_id & int & Hourly bucket (0-9) \\
open & float & First price in bucket \\
high & float & Maximum price in bucket \\
low & float & Minimum price in bucket \\
close & float & Last price in bucket \\
volume & int & Total contracts traded \\
vwap & float & Volume-weighted average price \\
\hline
\end{{tabular}}
\end{{center}}

\subsection{{Stage 2: Curve Construction}}

\textbf{{Purpose:}} Build continuous curves, calculate spreads, detect rolls.

\textbf{{Contract Ranking:}}
Contracts are labeled F1 through F12 based strictly on expiry date:
\begin{{itemize}}
    \item F1 = contract with nearest expiry that has DTE $>$ 0
    \item F2 = second-nearest expiry
    \item F3...F12 follow the same pattern
\end{{itemize}}

\textbf{{Spread Calculation:}}
\[
S1_{{raw}}(t) = F2(t) - F1(t) \quad\quad S1_{{pct}}(t) = \frac{{F2(t) - F1(t)}}{{F1(t)}}
\]

\textbf{{Roll Detection Thresholds:}}
\begin{{center}}
\begin{{tabular}}{{|l|c|l|}}
\hline
\textbf{{Phase}} & \textbf{{Volume Share}} & \textbf{{Interpretation}} \\
\hline
Pre-roll & $s < 25\%$ & F1 dominant \\
Roll Start & $s \geq 25\%$ & F2 gaining activity \\
Roll Peak & $s \geq 50\%$ & F2 overtakes F1 \\
Roll End & $s \geq 75\%$ & Transition complete \\
\hline
\end{{tabular}}
\end{{center}}

\subsection{{Stage 3: Analysis}}

\textbf{{Purpose:}} Generate summary statistics and event studies.

\textbf{{Analyses Performed:}}
\begin{{itemize}}
    \item \textbf{{DTE Lifecycle:}} Spread behavior by days-to-expiry bin
    \item \textbf{{Roll Event Study:}} Mean spread path $\pm$10 days around roll
    \item \textbf{{Diagnostics:}} Data quality checks, outlier counts
\end{{itemize}}

\subsection{{Stage 4: Backtesting}}

\textbf{{Purpose:}} Execute trading strategies and compute performance.

\textbf{{Strategy Definitions:}}

\begin{{center}}
\begin{{tabular}}{{|l|l|l|l|}}
\hline
\textbf{{Strategy}} & \textbf{{Entry Signal}} & \textbf{{Exit Signal}} & \textbf{{Default Direction}} \\
\hline
pre\_expiry & $\texttt{{exit\_dte}} < \texttt{{F1\_dte\_bdays}} \le \texttt{{entry\_dte}}$ & $\texttt{{F1\_dte\_bdays}} \le \texttt{{exit\_dte}}$ & Long spread \\
\hline
\end{{tabular}}
\end{{center}}

\textbf{{Output Schema (trade\_log):}}
\begin{{center}}
\begin{{tabular}}{{|l|l|l|}}
\hline
\textbf{{Column}} & \textbf{{Type}} & \textbf{{Description}} \\
\hline
trade\_id & int & Unique trade identifier \\
strategy & string & Strategy name (e.g., pre\_expiry) \\
entry\_date & datetime & Entry timestamp \\
exit\_date & datetime & Exit timestamp \\
direction & string & long/short \\
entry\_price & float & Spread price at entry \\
exit\_price & float & Spread price at exit \\
pnl\_gross & float & Gross P\&L before costs \\
cost & float & Transaction costs \\
pnl\_net & float & Net P\&L after costs \\
\hline
\end{{tabular}}
\end{{center}}

%==============================================================================
\section{{Configuration System}}
%==============================================================================

The pipeline is configured via YAML files. The default configuration is at
\texttt{{config/default.yaml}}.

\subsection{{Configuration Structure}}

\begin{{verbatim}}
# config/default.yaml
data:
  source_dir: /path/to/organized_data
  output_dir: data_parquet
  research_dir: research_outputs

pipeline:
  commodities: [HG]
  start_date: "2008-01-01"
  end_date: "2024-12-31"

backtest:
  slippage_ticks: 1
  commission_per_contract: 2.50
  tick_size: 0.0005
  tick_value: 12.50

strategies:
  pre_expiry:
    entry_dte: 5
    exit_dte: 1
    direction: long
\end{{verbatim}}

\subsection{{Key Parameters}}

{_config_table_to_latex()}

%==============================================================================
\section{{Transaction Cost Model}}
%==============================================================================

\subsection{{Cost Formula}}

Each spread trade involves 4 fills: entry F1, entry F2, exit F1, exit F2.
Total transaction cost per round-trip trade:

\[
\text{{Cost}} = 4 \times (\text{{slippage\_ticks}} \times \text{{tick\_value}} + \text{{commission}})
\]

\subsection{{Component Breakdown}}

\begin{{center}}
\begin{{tabular}}{{|l|l|l|}}
\hline
\textbf{{Component}} & \textbf{{Default (HG)}} & \textbf{{Description}} \\
\hline
Slippage & 1 tick = \$12.50 & Market impact per fill \\
Commission & \$2.50 & Broker fee per contract per side \\
Per-leg cost & \$15.00 & Slippage + commission \\
Round-trip & \$60.00 & 4 legs $\times$ \$15.00 \\
\hline
\end{{tabular}}
\end{{center}}

\subsection{{Cost Application}}

\begin{{itemize}}
    \item Costs are deducted from gross P\&L to compute net P\&L
    \item Slippage is applied adversely (buy at ask, sell at bid)
    \item All performance metrics (Sharpe, drawdown) use net P\&L
\end{{itemize}}

%==============================================================================
\section{{Execution Model}}
%==============================================================================

\subsection{{Fill Price Determination}}

To prevent look-ahead bias, fills use the next observation's spread mark
(bucket-close for bucket data; daily proxy close for daily data):
\begin{{itemize}}
    \item Signal generated at observation $t$ close
    \item Order executed at observation $t+1$ close (no same-observation fills)
    \item Slippage and commission deducted per fill (2 legs per side, 4 fills per round-trip)
\end{{itemize}}

\subsection{{Position Sizing}}

\begin{{itemize}}
    \item Default: 1 spread unit per signal
    \item Spread unit = 1 contract F1 vs. 1 contract F2
    \item No leverage or margin calculations in current implementation
\end{{itemize}}

\subsection{{Look-Ahead Bias Prevention}}

\begin{{enumerate}}
    \item Entry signals use data available at signal time only
    \item Execution uses the next observation (no same-observation fills)
    \item DTE calculations use expiry dates known at trade date
    \item Roll detection uses only past volume data
\end{{enumerate}}

%==============================================================================
\section{{Output Artifacts}}
%==============================================================================

The pipeline produces outputs in the following directory structure:

\begin{{verbatim}}
metadata/
  contract_specs.json       - Contract specs (tick size/value, months, etc.)
  <SYMBOL>_expiry.parquet   - Expiry schedule (business-day calendar)

data_parquet/
  buckets/<SYMBOL>/         - Optional contract-level bucket aggregates
  daily/<SYMBOL>/           - Contract-level daily bars
  qc/                       - QC summaries
  curve/<SYMBOL>/           - Curve panels (F1-F12) at bucket and daily frequency
  spreads/<SYMBOL>/         - Spread panels (S1..S11), near/far IDs, DTE
  roll_events/<SYMBOL>/     - Roll shares + detected events
  trades/<SYMBOL>/          - Backtest trade logs, equity, summary

research_outputs/
  tables/             - Summary statistics (parquet)
  figures/            - Visualizations (PDF)
  report/             - Generated reports (PDF)
\end{{verbatim}}

%==============================================================================
\section{{Testing and Validation}}
%==============================================================================

\subsection{{Data Quality Checks}}

The pipeline performs automated validation:
\begin{{itemize}}
    \item \textbf{{OHLC Consistency:}} Verify High $\geq$ Open, Close, Low
    \item \textbf{{Timestamp Monotonicity:}} Check strictly increasing timestamps
    \item \textbf{{Outlier Detection:}} Z-score $>$ 3 flagged for review
    \item \textbf{{Expiry Constraints:}} Verify DTE $>$ 0 for active contracts
    \item \textbf{{Volume Sanity:}} Flag zero-volume buckets
\end{{itemize}}

\subsection{{Backtest Verification}}

\begin{{itemize}}
    \item Trade log timestamps verified against signal generation
    \item P\&L reconciliation: $\text{{gross}} - \text{{cost}} = \text{{net}}$
    \item Equity curve final value matches sum of trade P\&L
    \item No overlapping positions (one trade at a time per strategy)
\end{{itemize}}

\subsection{{Edge Case Handling}}

\begin{{itemize}}
    \item Missing buckets filled with NaN, excluded from analysis
    \item Contract gaps (no trading) handled via forward-fill
    \item Roll events near expiry use accelerated detection
\end{{itemize}}

%==============================================================================
\section{{Reproducibility Checklist}}
%==============================================================================

To reproduce the analysis:

\begin{{enumerate}}
    \item \textbf{{Environment Setup}}
    \begin{{verbatim}}
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    \end{{verbatim}}

    \item \textbf{{Data Requirements}}
    \begin{{itemize}}
        \item Raw tick data in vendor format at configured path
        \item Minimum: 2008-2024 for copper (HG)
        \item Approximately 50GB raw, 5GB processed
    \end{{itemize}}

    \item \textbf{{Pipeline Execution}}
    \begin{{verbatim}}
    # Full pipeline
    python -m futures_curve.cli run \
        --config config/default.yaml \
        --commodities HG

    # Reports only (after pipeline)
    python -m futures_curve.cli report \
        --symbol HG --report-type both
    \end{{verbatim}}

    \item \textbf{{Verification}}
    \begin{{itemize}}
        \item Check \texttt{{data\_parquet/}} for parquet outputs
        \item Check \texttt{{research\_outputs/figures/}} for PDF plots
        \item Check \texttt{{research\_outputs/report/}} for final reports
    \end{{itemize}}
\end{{enumerate}}

%==============================================================================
\appendix
\section{{Glossary}}
%==============================================================================

{_glossary_to_latex()}

%==============================================================================
\section{{CLI Reference}}
%==============================================================================

{_cli_reference_to_latex()}

\end{{document}}
"""

    tex_path = paths.report_dir / f"{symbol}_Technical_Implementation_Report.tex"
    tex_path.write_text(tex, encoding="utf-8")

    if compile_pdf:
        _compile_latex(tex_path, cwd=paths.report_dir)

    return tex_path


def build_analysis_report(
    symbol: str,
    data_dir: str | Path = "data_parquet",
    research_dir: str | Path = "research_outputs",
    compile_pdf: bool = True,
    generate_figures: bool = True,
) -> Path:
    """Build the Analysis Results Report.

    Audience: Research/analysis consumers
    Purpose: Present findings with professional visualizations
    """
    paths = build_report_paths(data_dir, research_dir)
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures (optional: skipped for unit tests)
    if generate_figures:
        from .visualization.analysis_plots import AnalysisVisualizer
        from .visualization.backtest_plots import BacktestVisualizer

        AnalysisVisualizer(paths.figures_dir).generate_all(
            symbol, paths.tables_dir, data_dir=paths.data_dir
        )
        BacktestVisualizer(paths.figures_dir).generate_all(
            symbol, paths.data_dir / "trades" / symbol
        )

    # Load key tables
    diagnostics = _read_parquet_optional(paths.tables_dir / f"{symbol}_diagnostics.parquet")
    dte_profile = _read_parquet_optional(paths.tables_dir / f"{symbol}_dte_profile.parquet")
    backtest_summary = _read_parquet_optional(paths.data_dir / "trades" / symbol / "summary.parquet")
    spread_panel = _read_parquet_optional(paths.data_dir / "spreads" / symbol / "spreads_panel.parquet")
    qc_stats = _read_parquet_optional(paths.data_dir / "qc" / f"{symbol}_qc.parquet")
    pre_expiry_sweep = _read_parquet_optional(paths.data_dir / "trades" / symbol / "pre_expiry_sweep.parquet")
    pre_expiry_stop_loss = _read_parquet_optional(
        paths.data_dir / "trades" / symbol / "pre_expiry_stop_loss_sensitivity.parquet"
    )

    # Load sample trades for appendix
    sample_trades = _get_sample_trades(paths.data_dir / "trades" / symbol, n=15)

    # Report provenance (helps prevent “stale PDF” confusion when multiple runs exist)
    from datetime import datetime, timezone

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_path = paths.data_dir / "trades" / symbol / "summary.parquet"
    summary_mtime_utc = (
        datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if summary_path.exists()
        else "N/A"
    )

    # Bucketing table (exchange time, US/Central)
    bucket_tbl = _bucket_table()
    # Prevent overfull hboxes from long descriptions: wrap the last column.
    bucket_tex = _df_to_latex(bucket_tbl, floatfmt=".0f", column_format="rllp{8cm}")

    # Coverage / data facts (prefer actual artifacts over hard-coded claims)
    if spread_panel is not None and not spread_panel.empty and "trade_date" in spread_panel.columns:
        td_min = pd.to_datetime(spread_panel["trade_date"]).min().date()
        td_max = pd.to_datetime(spread_panel["trade_date"]).max().date()
        n_trade_days = int(pd.to_datetime(spread_panel["trade_date"]).nunique())
    else:
        td_min = None
        td_max = None
        n_trade_days = None

    contracts_processed = (
        int(qc_stats.iloc[0]["contracts_processed"])
        if qc_stats is not None and not qc_stats.empty and "contracts_processed" in qc_stats.columns
        else None
    )
    total_minute_rows = (
        int(qc_stats.iloc[0]["total_minute_rows"])
        if qc_stats is not None and not qc_stats.empty and "total_minute_rows" in qc_stats.columns
        else None
    )

    analysis_start_str = td_min.isoformat() if td_min is not None else "N/A"
    analysis_end_str = td_max.isoformat() if td_max is not None else "N/A"
    trade_days_str = f"{n_trade_days:,}" if n_trade_days is not None else "N/A"
    contracts_str = f"{contracts_processed:,}" if contracts_processed is not None else "N/A"
    minute_rows_str = f"{total_minute_rows:,}" if total_minute_rows is not None else "N/A"

    # Backtest performance table with cost decomposition (derived from trade logs).
    def _cost_breakdown(trades_dir: Path) -> pd.DataFrame:
        records: list[dict] = []
        for p in sorted(trades_dir.glob("*_trades.parquet")):
            strat = p.stem.replace("_trades", "")
            df = pd.read_parquet(p)
            if df.empty or "pnl" not in df.columns:
                continue
            closed = df[df.get("status", "closed").isin(["closed", "rolled"])].copy()
            net = float(closed["pnl"].sum())
            total_slip = float(closed["slippage_cost"].sum()) if "slippage_cost" in closed.columns else 0.0
            total_comm = float(closed["commission_cost"].sum()) if "commission_cost" in closed.columns else 0.0
            total_costs = total_slip + total_comm
            gross = net + total_costs
            n = int(len(closed))
            records.append(
                {
                    "strategy": strat,
                    "gross_pnl": gross,
                    "total_costs": total_costs,
                    "net_pnl": net,
                    "total_slippage": total_slip,
                    "total_commission": total_comm,
                    "cost_per_trade": (total_costs / n) if n else 0.0,
                }
            )
        return pd.DataFrame(records)

    trades_dir = paths.data_dir / "trades" / symbol
    costs_df = _cost_breakdown(trades_dir)
    perf_df = None
    if backtest_summary is not None and not backtest_summary.empty:
        perf_df = backtest_summary.copy()
        if costs_df is not None and not costs_df.empty:
            perf_df = perf_df.merge(costs_df, on="strategy", how="left")

    if perf_df is not None and not perf_df.empty:
        perf_display = perf_df.copy()
        perf_display["strategy"] = perf_display["strategy"].str.upper()
        cols = [
            "strategy",
            "total_trades",
            "win_rate",
            "gross_pnl",
            "total_costs",
            "total_pnl",
            "sharpe_ratio",
            "max_drawdown_pct",
            "profit_factor",
        ]
        cols = [c for c in cols if c in perf_display.columns]
        perf_display = perf_display[cols]
        perf_tex = _df_to_latex(perf_display, floatfmt=".2f")
    else:
        perf_tex = "\\emph{(No backtest summary found)}"

    def _fmt_signed_money(x: float | int | None) -> str:
        if x is None:
            return "N/A"
        try:
            val = float(x)
        except Exception:
            return "N/A"
        sign = "+" if val >= 0 else "-"
        # LaTeX: escape dollar sign to avoid accidental math-mode / overfull hboxes.
        return f"{sign}\\${abs(val):,.2f}"

    def _fmt_float(x: float | int | None, ndigits: int = 2) -> str:
        if x is None:
            return "N/A"
        try:
            return f"{float(x):.{ndigits}f}"
        except Exception:
            return "N/A"

    if perf_df is not None and not perf_df.empty:
        pre_expiry_metrics = (
            perf_df[perf_df["strategy"] == "pre_expiry"].iloc[0].to_dict()
            if (perf_df["strategy"] == "pre_expiry").any()
            else {}
        )
    else:
        pre_expiry_metrics = {}

    pre_expiry_net_str = _fmt_signed_money(pre_expiry_metrics.get("total_pnl"))
    pre_expiry_sharpe_str = _fmt_float(pre_expiry_metrics.get("sharpe_ratio"))

    # Stop-loss sensitivity tables (robustness section)
    def _stop_loss_tex(df: Optional[pd.DataFrame]) -> str:
        if df is None or df.empty:
            return "\\emph{(No stop-loss sensitivity results found)}"
        out = df.copy()
        keep = [
            "stop_loss_usd",
            "total_trades",
            "win_rate",
            "gross_pnl",
            "total_costs",
            "total_pnl",
            "sharpe_ratio",
            "max_drawdown_pct",
        ]
        keep = [c for c in keep if c in out.columns]
        out = out[keep]
        return _df_to_latex(out, floatfmt=".2f")

    pre_expiry_sl_tex = _stop_loss_tex(pre_expiry_stop_loss)

    # Pre-expiry sweep table (top rows)
    if pre_expiry_sweep is not None and not pre_expiry_sweep.empty:
        sweep_fmt = pre_expiry_sweep.copy()
        keep = [
            "entry_dte",
            "exit_dte",
            "total_trades",
            "win_rate",
            "total_pnl",
            "sharpe_ratio",
            "max_drawdown_pct",
        ]
        keep = [c for c in keep if c in sweep_fmt.columns]
        sweep_fmt = sweep_fmt[keep].head(15)
        sweep_tex = _df_to_latex(sweep_fmt, floatfmt=".2f")
    else:
        sweep_tex = "\\emph{(No sweep results found)}"

    if dte_profile is not None and not dte_profile.empty:
        dte_fmt = dte_profile.copy()
        if "mean" in dte_fmt.columns:
            dte_fmt["mean_pct"] = dte_fmt["mean"] * 100
        if "median" in dte_fmt.columns:
            dte_fmt["median_pct"] = dte_fmt["median"] * 100
        dte_tex = _df_to_latex(dte_fmt, floatfmt=".3f")
    else:
        dte_tex = "\\emph{(No DTE profile found)}"

    diag_tex = _df_to_latex(diagnostics, floatfmt=".0f") if diagnostics is not None else "\\emph{(No diagnostics)}"
    bt_tex = perf_tex

    # Sample trades for appendix
    if sample_trades is not None and not sample_trades.empty:
        trades_tex = _df_to_latex(sample_trades, floatfmt=".4f")
    else:
        trades_tex = "\\emph{(No trade records available)}"

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\usepackage{{float}}
\usepackage{{xcolor}}
\usepackage{{tocloft}}
\usepackage{{subcaption}}
\usepackage{{enumitem}}
\usepackage{{longtable}}

\definecolor{{primary}}{{HTML}}{{2C3E50}}
\definecolor{{positive}}{{HTML}}{{27AE60}}
\definecolor{{negative}}{{HTML}}{{C0392B}}

\hypersetup{{
    colorlinks=true,
    linkcolor=primary,
    urlcolor=primary,
}}

\title{{\textbf{{{symbol} Futures Curve Analysis}}\\[0.5em]\Large Analysis Results Report}}
\author{{futures\_curve Pipeline}}
\date{{\today}}

\begin{{document}}
\maketitle
\noindent\textbf{{Run provenance}}\\
\begin{{itemize}}[noitemsep]
  \item data\_dir: \verb|{paths.data_dir.resolve()}|
  \item research\_dir: \verb|{paths.research_dir.resolve()}|
  \item generated (UTC): {generated_utc}
  \item trades summary mtime (UTC): {summary_mtime_utc}
\end{{itemize}}
\medskip
\tableofcontents
\newpage

%==============================================================================
\section{{Executive Summary}}
%==============================================================================

This report documents a reproducible analysis of the \textbf{{{symbol}}} front calendar spread
($S1 = F2 - F1$) and evaluates an expiry-anchored \texttt{{pre\_expiry}} trading rule under an explicit execution
and transaction-cost model.

\subsection{{Key Findings}}

\begin{{itemize}}
    \item \textbf{{Data coverage:}} {contracts_str} contracts; {minute_rows_str} 1-minute bars; {analysis_start_str} to {analysis_end_str} ({trade_days_str} trade dates).
    \item \textbf{{Market structure:}} {symbol} is often in contango, with measurable variation across the contract lifecycle (DTE) and roll activity.
    \item \textbf{{Profitability (net of costs):}} \texttt{{pre\_expiry}} net P\&L {pre_expiry_net_str}, Sharpe {pre_expiry_sharpe_str} (this run).
    \item \textbf{{Interpretation:}} Results are conditional on the execution proxy (bucket/daily closes) and the transaction cost model; sensitivity analyses are provided in Sections 9--10.
\end{{itemize}}

\subsection{{How to read this report}}

Sections 2--6 define the data transformations, bucketing, and feature construction. Sections 7--8 define the strategies and show baseline backtest results. Sections 9--10 provide robustness checks (transaction costs, stop-loss sensitivity) and risk diagnostics.

%==============================================================================
\section{{Data and Preprocessing}}
%==============================================================================

\subsection{{Raw data format}}

Input files are headerless 1-minute OHLCV TXT/CSV in the form:
\begin{{verbatim}}
(timestamp, open, high, low, close, volume)
\end{{verbatim}}

Timestamps are typically \emph{{naive}} (no timezone). The pipeline infers the raw timezone by choosing the conversion to exchange time that minimizes activity during the CME maintenance hour (16:00--16:59 CT).

\subsection{{Exchange time and trade date}}

All timestamps are converted to \textbf{{US/Central}} (CME exchange time). A \textbf{{trade date}} is assigned using the CME Globex boundary:
\begin{{itemize}}
  \item If timestamp $\ge$ 17:00 CT, trade date = next calendar date.
  \item Otherwise, trade date = same calendar date.
\end{{itemize}}

\subsection{{Bucketing (hourly session schema)}}

Minute bars are aggregated into a 10-bucket schema per trade date. Buckets are defined so that the trade-date boundary (17:00 CT) is also the start of bucket 8. Bucket 0 (16:00--16:59 CT) is treated as QC-only.

\begin{{table}}[H]
\centering
{bucket_tex}
\caption{{Bucket schema in exchange time (US/Central). Note: buckets 8 and 9 occur on the \emph{{prior}} calendar day but belong to the next trade date.}}
\label{{tab:bucket_schema}}
\end{{table}}

\subsection{{Aggregation outputs}}

The pipeline produces:
\begin{{itemize}}
  \item Contract-level bucket OHLCV (Stage 1)
  \item Deterministic curve panel (F1..F12) and spreads (Stage 2)
  \item Lifecycle profiles, roll event studies, and diagnostics (Stage 3)
  \item Backtests with trade logs and equity curves (Stage 4)
\end{{itemize}}

%==============================================================================
\section{{Data Cleaning and Quality}}
%==============================================================================

\subsection{{Validation Checks Performed}}

\begin{{itemize}}
    \item \textbf{{OHLC Consistency:}} Verified High $\geq$ max(Open, Close) and
          Low $\leq$ min(Open, Close) for all bars
    \item \textbf{{Outliers:}} Large spread moves are flagged for review (retained unless explicitly excluded for a specific statistic)
    \item \textbf{{Expiry constraint:}} Verified DTE $>$ 0 for all eligible F1 labels (no expired-contract trading)
    \item \textbf{{Missing data:}} Spread observations require both legs to have a print in the bucket; missing prints propagate to NA spreads (no forward-fill)
\end{{itemize}}

\subsection{{Diagnostics Summary}}

\begin{{table}}[H]
\centering
{diag_tex}
\caption{{Data quality diagnostics from pipeline validation checks.}}
\end{{table}}

\subsection{{Outlier Handling}}

Outliers (Z-score $>$ 3) were handled as follows:
\begin{{itemize}}
    \item Retained in the dataset for transparency
    \item Flagged in diagnostics for researcher review
    \item Where a statistic is sensitive to extremes, robustness checks are reported explicitly
\end{{itemize}}

%==============================================================================
\section{{Feature Engineering}}
%==============================================================================

\subsection{{Calendar Spread Calculation}}

The front calendar spread (S1) is calculated as:
\[
S1_{{raw}}(t) = F2(t) - F1(t)
\]

where F1 and F2 are the front and second-month contracts respectively.

\subsection{{Spread Normalization}}

For cross-commodity and cross-time comparisons, we normalize:
\[
S1_{{pct}}(t) = \frac{{F2(t) - F1(t)}}{{F1(t)}}
\]

This expresses the spread as a fraction of the front-month price (multiply by 100 for percentage points when desired).

\subsection{{Days-to-Expiry (DTE)}}

DTE is calculated as business days remaining until F1 contract expiration:
\begin{{itemize}}
    \item Computed using the CME trading calendar (weekends and exchange holidays excluded)
    \item Business-day DTE (\texttt{{F1\_dte\_bdays}}) is computed on the trade-date axis
    \item Hour-based DTE (\texttt{{F1\_dte\_hours}}) is computed from bucket-end timestamps (for diagnostics)
\end{{itemize}}

\subsection{{Roll Detection}}

Roll timing is detected via F2 volume share:
\[
s(t) = \frac{{V_{{F2}}(t)}}{{V_{{F1}}(t) + V_{{F2}}(t)}}
\]

Roll phases: Start ($s \geq 25\%$), Peak ($s \geq 50\%$), End ($s \geq 75\%$).

%==============================================================================
\section{{Spread Characteristics}}
%==============================================================================

\subsection{{Term Structure Analysis}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_term_structure.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_term_structure.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{{symbol} term structure snapshot showing price curve across contracts F1-F12.
    Upward-sloping curve indicates contango; downward-sloping indicates backwardation.}}
\end{{figure}}

\textbf{{Interpretation:}} The term structure snapshot captures the price relationship
across contract months at a single point in time. For {symbol}, the curve typically
exhibits contango, reflecting storage costs and convenience yield dynamics.

\subsection{{DTE Profile}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_dte_profile.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_dte_profile.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{Spread characteristics by days-to-expiry showing how contango/backwardation
    frequency and mean spread evolve across the contract lifecycle.}}
\end{{figure}}

\begin{{table}}[H]
\centering
{dte_tex}
\caption{{Spread statistics by DTE bin. Values in percentage terms where noted.}}
\end{{table}}

\textbf{{Interpretation:}} The DTE profile reveals systematic patterns in spread
behavior as contracts approach expiry. Near-expiry periods (DTE $<$ 10) often show
increased volatility due to roll activity.

\subsection{{Contango/Backwardation Analysis}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_contango_timeline.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_contango_timeline.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{Timeline of contango (green) and backwardation (red) states across the
    sample period. Contango predominates but backwardation episodes occur.}}
\end{{figure}}

%==============================================================================
\section{{Intraday Patterns}}
%==============================================================================

Bucket-level aggregation allows a simple view of how the spread behaves by
time-of-day and calendar month. These plots are descriptive only (no costs).

\subsection{{Bucket-Level Analysis}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_bucket_seasonality.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_bucket_seasonality.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{Mean spread by trading bucket and calendar month, showing intraday and
    seasonal interaction effects.}}
\end{{figure}}

%==============================================================================
\section{{Roll Dynamics}}
%==============================================================================

\subsection{{Roll Event Study}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_roll_event_study.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_roll_event_study.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{Mean spread behavior around roll events with 95\% confidence bands.
    Day 0 = roll peak (F2 volume share $>$ 50\%).}}
\end{{figure}}

\textbf{{Interpretation:}} The event study aligns all roll events at day 0 (roll peak)
and computes the average spread path. Confidence bands indicate the variability
across roll events.

\subsection{{Volume Share Evolution}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_volume_share.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_volume_share.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{F2 volume share as a function of F1 DTE (business days to expiry),
    showing the characteristic S-curve transition pattern during contract rolls.}}
\end{{figure}}

%==============================================================================
\section{{Strategy Backtesting}}
%==============================================================================

\subsection{{Strategy Definitions}}

\begin{{center}}
\begin{{tabular}}{{|l|p{{4.8cm}}|p{{4.8cm}}|l|}}
\hline
\textbf{{Strategy}} & \textbf{{Entry Rule}} & \textbf{{Exit Rule}} & \textbf{{Direction}} \\
\hline
PRE\_EXPIRY & First signal when \texttt{{exit\_dte}} $< \texttt{{F1\_dte\_bdays}} \le$ \texttt{{entry\_dte}} (executed next observation) & First signal when $\texttt{{F1\_dte\_bdays}} \le$ \texttt{{exit\_dte}} (executed next observation) & Long $S1$ \\
\hline
\end{{tabular}}
\end{{center}}

\subsection{{Configuration Parameters}}

\begin{{center}}
\begin{{tabular}}{{|l|c|l|}}
\hline
\textbf{{Parameter}} & \textbf{{Value}} & \textbf{{Description}} \\
\hline
Slippage & 1 tick & \$12.50 per fill (HG) \\
Commission & \$2.50 & Per contract per side \\
Position size & 1 spread & 1 F1 vs. 1 F2 contract \\
Round-trip cost & \$60.00 & 4 fills $\times$ (\$12.50 + \$2.50) \\
\hline
\end{{tabular}}
\end{{center}}

\subsection{{Pre-Expiry Parameter Sweep}}

\begin{{table}}[H]
\centering
{sweep_tex}
\caption{{Top \texttt{{pre\_expiry}} entry/exit windows ranked by net P\&L (after costs).}}
\end{{table}}

\subsection{{Performance Summary}}

\begin{{table}}[H]
\centering
{bt_tex}
\caption{{Strategy-level performance metrics. \texttt{{total\_pnl}} is net of transaction costs; \texttt{{gross\_pnl}} and \texttt{{total\_costs}} are derived from the trade logs.}}
\end{{table}}

\subsection{{Equity Curves}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_pre_expiry_equity.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_pre_expiry_equity.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{\texttt{{pre\_expiry}} strategy equity curve with drawdown overlay showing cumulative
    net P\&L over time.}}
\end{{figure}}

\subsection{{Strategy Comparison}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_strategy_comparison.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_strategy_comparison.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{Comparison of total P\&L and Sharpe ratio across strategies.}}
\end{{figure}}

\subsection{{P\&L Distribution}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_pre_expiry_pnl_dist.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_pre_expiry_pnl_dist.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{Distribution of trade P\&L for the \texttt{{pre\_expiry}} strategy showing win/loss
    magnitude asymmetry.}}
\end{{figure}}

\subsection{{Monthly Returns}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_pre_expiry_monthly_heatmap.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_pre_expiry_monthly_heatmap.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{Year-by-month P\&L heatmap for the \texttt{{pre\_expiry}} strategy showing performance
    consistency across time.}}
\end{{figure}}

%==============================================================================
\section{{Sensitivity and Robustness}}
%==============================================================================

\subsection{{Transaction Cost Sensitivity}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_pre_expiry_cost_sensitivity.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_pre_expiry_cost_sensitivity.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{\texttt{{pre\_expiry}} strategy profitability sensitivity to slippage and commission assumptions.
    Vertical line indicates baseline assumption.}}
\end{{figure}}

\subsection{{Break-Even Analysis}}

Transaction costs are a critical determinant of strategy viability. The cost
sensitivity figure shows:
\begin{{itemize}}
    \item Net P\&L across a range of cost assumptions
    \item Break-even point where strategy becomes unprofitable
    \item Margin of safety relative to realistic cost estimates
\end{{itemize}}

\textbf{{Discussion:}} The cost sensitivity plot shows how robust the strategy is to plausible slippage/commission assumptions. A strategy that remains profitable under conservative costs is more credible.

\subsection{{Stop-Loss Sensitivity (robustness check)}}

Stop losses are \emph{{not}} part of the baseline definition. Here we evaluate a simple stop-loss rule as a robustness check:
if the \textbf{{gross}} mark-to-market P\&L (excluding costs) falls below \texttt{{-stop\_loss\_usd}} at an observation close, the position is closed.

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_pre_expiry_stop_loss_sensitivity.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_pre_expiry_stop_loss_sensitivity.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{\texttt{{pre\_expiry}} stop-loss sensitivity.}}
\end{{figure}}

\begin{{table}}[H]
\centering
{pre_expiry_sl_tex}
\caption{{\texttt{{pre\_expiry}} stop-loss sensitivity table (net of costs).}}
\end{{table}}

%==============================================================================
\section{{Risk Analysis}}
%==============================================================================

\subsection{{Rolling Performance}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_pre_expiry_rolling_performance.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_pre_expiry_rolling_performance.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{\texttt{{pre\_expiry}} rolling Sharpe ratio and win rate (20-trade window).}}
\end{{figure}}

\subsection{{Cumulative Monthly P\&L}}

\begin{{figure}}[H]
\centering
\IfFileExists{{../figures/{symbol}_pre_expiry_cumulative_monthly.pdf}}{{
    \includegraphics[width=0.95\textwidth]{{../figures/{symbol}_pre_expiry_cumulative_monthly.pdf}}
}}{{
    \emph{{(Figure not available)}}
}}
\caption{{\texttt{{pre\_expiry}} monthly P\&L breakdown with cumulative total.}}
\end{{figure}}

%==============================================================================
\section{{Threats to Validity}}
%==============================================================================

\subsection{{Look-Ahead Bias}}

\textbf{{Mitigation:}} All signals use only data available at signal time. The backtester enforces a one-observation delay between signal and execution (no same-observation fills).

\textbf{{Residual risk:}} Execution prices are observation closes (bucket close / daily proxy) rather than bid/ask executable prices. This can overstate realizability in fast markets.

\subsection{{Transaction Cost Assumptions}}

\textbf{{Concern:}} Baseline costs may not reflect actual execution, especially
during volatile periods.

\textbf{{Mitigation:}} Sensitivity analysis tests a wide range of cost assumptions.
Results should be interpreted alongside these robustness checks.

\subsection{{Limited Strategy Variants}}

\textbf{{Concern:}} Parameter sweeps over multiple entry/exit windows can overfit
the historical sample.

\textbf{{Mitigation:}} Prefer stable regions in the sweep surface (not a single best cell),
and validate chosen parameters out-of-sample (walk-forward or holdout period).

\subsection{{Single Commodity Focus}}

\textbf{{Concern:}} Results may not generalize to other commodities.

\textbf{{Mitigation:}} The pipeline architecture supports multi-commodity analysis.
Future work should test strategies on a broader universe.

\subsection{{Survivorship Bias}}

\textbf{{Concern:}} Analysis includes only contracts that completed their lifecycle.

\textbf{{Assessment:}} Not applicable for futures (contracts expire per schedule,
not delisted due to performance).

%==============================================================================
\section{{Conclusions and Recommendations}}
%==============================================================================

\subsection{{Summary of Findings}}

\begin{{enumerate}}
    \item \textbf{{Spread Dynamics:}} {symbol} calendar spreads exhibit systematic
          patterns related to contract lifecycle (DTE) and roll dynamics
    \item \textbf{{Roll Behavior:}} Volume share transition provides a reliable
          signal for roll timing
    \item \textbf{{Strategy Viability ({symbol}):}} \texttt{{pre\_expiry}} performance depends on the chosen entry/exit window and the assumed execution costs; sweep results highlight the best in-sample windows.
\end{{enumerate}}

\subsection{{Strategy Viability Assessment}}

Based on the backtest results:
\begin{{itemize}}
    \item \textbf{{Pre-Expiry Strategy:}} Anchored to F1 expiry (business-day DTE); performance is evaluated net of costs and stress-tested with cost and stop-loss sensitivity.
\end{{itemize}}

\subsection{{Recommended Next Steps}}

\begin{{enumerate}}
    \item \textbf{{Out-of-Sample Testing:}} Reserve recent data for validation
    \item \textbf{{Multi-Commodity Extension:}} Test on GC, CL, and other metals
    \item \textbf{{Parameter Sensitivity:}} Explore \texttt{{pre\_expiry}} entry/exit DTE windows (and look for stable regions, not a single best cell)
    \item \textbf{{Regime Analysis:}} Condition strategies on volatility regime
    \item \textbf{{Live Paper Trading:}} Forward test with simulated execution
\end{{enumerate}}

%==============================================================================
\appendix
\section{{Parameter Reference}}
%==============================================================================

This appendix documents the configuration parameters used for all analyses in
this report.

\subsection{{Pipeline Configuration}}

\begin{{verbatim}}
data_dir: {paths.data_dir.resolve()}
research_dir: {paths.research_dir.resolve()}
\end{{verbatim}}

\subsection{{Backtest Configuration}}

\begin{{verbatim}}
Execution:
  slippage_ticks: 1
  commission_per_contract: 2.50
  tick_size: 0.0005
  tick_value: 12.50
  initial_capital: 100000

pre_expiry strategy:
  entry: exit_dte < F1_dte_bdays <= entry_dte
  exit:  F1_dte_bdays <= exit_dte
  direction: long S1

Sweep (optional):
  entry_dte: 2..10
  exit_dte: 0..3
  objective: maximize net total_pnl
\end{{verbatim}}

%==============================================================================
\section{{Sample Trade Records}}
%==============================================================================

First 15 trades from the backtest trade log:

\begin{{table}}[H]
\centering
\small
{trades_tex}
\caption{{Sample trade records showing entry/exit dates, direction, prices, and net P\&L.}}
\end{{table}}

%==============================================================================
\section{{Figure Index}}
%==============================================================================

\listoffigures

%==============================================================================
\section{{Methodology Notes}}
%==============================================================================

\begin{{itemize}}
    \item \textbf{{Exchange Time:}} All timestamps are in US/Central (CME).
    \item \textbf{{Trade Date Boundary:}} 17:00 CT marks start of each trade date.
    \item \textbf{{Contract Labels:}} F1--F12 ranked strictly by expiry (not by volume).
    \item \textbf{{Spread:}} $S1 = F2 - F1$ in price units; normalized as $(F2-F1)/F1$.
    \item \textbf{{Transaction Costs:}} Modeled per-leg per-side (4 fills per round trip).
    \item \textbf{{Sharpe:}} Annualized using equity-curve timestamp spacing (data-driven annualization factor).
\end{{itemize}}

\end{{document}}
"""

    tex_path = paths.report_dir / f"{symbol}_Analysis_Results_Report.tex"
    tex_path.write_text(tex, encoding="utf-8")

    if compile_pdf:
        _compile_latex(tex_path, cwd=paths.report_dir)

    return tex_path


def build_pipeline_report(
    symbol: str,
    data_dir: str | Path = "data_parquet",
    research_dir: str | Path = "research_outputs",
    compile_pdf: bool = True,
    report_type: Literal["technical", "analysis", "both"] = "both",
) -> Path | tuple[Path, Path]:
    """Build pipeline reports.

    Args:
        symbol: Commodity symbol (e.g., "HG")
        data_dir: Path to parquet data directory
        research_dir: Path to research outputs directory
        compile_pdf: Whether to compile LaTeX to PDF
        report_type: Which report(s) to generate:
            - "technical": Technical Implementation Report only
            - "analysis": Analysis Results Report only
            - "both": Both reports (default)

    Returns:
        Path to .tex file(s) generated. If "both", returns tuple of paths.
    """
    if report_type == "technical":
        return build_technical_report(symbol, data_dir, research_dir, compile_pdf)
    elif report_type == "analysis":
        return build_analysis_report(symbol, data_dir, research_dir, compile_pdf)
    else:  # both
        tech_path = build_technical_report(symbol, data_dir, research_dir, compile_pdf)
        analysis_path = build_analysis_report(symbol, data_dir, research_dir, compile_pdf)
        return tech_path, analysis_path


def _compile_latex(tex_path: Path, cwd: Optional[Path] = None) -> None:
    """Compile a TeX file with pdflatex (twice for TOC)."""
    cwd = cwd or tex_path.parent
    cwd = Path(cwd)
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    # Run twice to resolve TOC and references
    subprocess.run(cmd, cwd=str(cwd), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run(cmd, cwd=str(cwd), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Hygiene: remove intermediate build artifacts, keep only .tex and .pdf.
    stem = tex_path.stem
    keep = {f"{stem}.tex", f"{stem}.pdf"}
    for p in cwd.glob(f"{stem}.*"):
        if p.name in keep:
            continue
        try:
            p.unlink()
        except FileNotFoundError:
            pass

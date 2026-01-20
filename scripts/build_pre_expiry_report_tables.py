from __future__ import annotations

"""Build LaTeX tables for the pre-expiry DTE report.

This keeps the report sources in `reports/` small and readable while still
pulling the latest numbers from the generated outputs in `output/`.
"""

import argparse
from pathlib import Path

import pandas as pd


def _latex_escape(text: str) -> str:
    # Minimal escaping for tabular content.
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


def _fmt_float(x: float, *, digits: int = 6) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return f"{float(x):.{digits}f}"


def _fmt_int(x: float | int) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return f"{int(x)}"


def _fmt_money(x: float) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return f"{float(x):,.2f}"


def _table_header(cols: list[str]) -> str:
    spec = "l" + "r" * (len(cols) - 1)
    header = " & ".join(cols) + r" \\"
    return "\n".join(
        [
            r"\begin{tabular}{" + spec + r"}",
            r"\toprule",
            header,
            r"\midrule",
        ]
    )


def _table_footer() -> str:
    return "\n".join([r"\bottomrule", r"\end{tabular}"])


def _render_rows(rows: list[list[str]]) -> str:
    return "\n".join([" & ".join(r) + r" \\" for r in rows])


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def build_tables(*, output_root: Path) -> str:
    # Top window candidates (ranked on full sample)
    ranked_windows = _read_csv(output_root / "robustness" / "baseline_ranked_windows.csv").head(10)

    # Walk-forward OOS (expanding selection)
    wf_summary = _read_csv(output_root / "walkforward" / "walkforward_oos_summary.csv")

    # Fixed-rule robustness (pooled comparison)
    robust_summary = _read_csv(output_root / "robustness" / "summary.csv")

    # Baseline fixed-rule year-by-year (expiry year buckets)
    baseline_by_year = _read_csv(
        output_root / "robustness" / "baseline_us_vwap_shift0_cost1" / "summary_by_expiry_year.csv"
    )

    parts: list[str] = []

    # Table: Top ranked windows
    parts.append(r"\section*{Top ranked windows (in-sample; costed)}")
    parts.append(r"\footnotesize")
    cols = ["Regime", "Dir", "Entry", "Exit", "N", "Net mean (USD/ct)", "Net t"]
    parts.append(_table_header(cols))
    rows = []
    for _, row in ranked_windows.iterrows():
        rows.append(
            [
                _latex_escape(str(row.get("regime", ""))),
                _latex_escape(str(row.get("direction", ""))),
                _fmt_int(row.get("entry_dte", "")),
                _fmt_int(row.get("exit_dte", "")),
                _fmt_int(row.get("n", "")),
                _fmt_money(row.get("net_mean_usd", float("nan"))),
                _fmt_float(row.get("net_t", float("nan")), digits=2),
            ]
        )
    parts.append(_render_rows(rows))
    parts.append(_table_footer())
    parts.append(r"\normalsize")

    # Table: Walk-forward OOS summary
    parts.append(r"\section*{Walk-forward out-of-sample summary}")
    parts.append(r"\footnotesize")
    cols = ["Year", "Regime", "Dir", "Entry", "Exit", "N", "Net mean (USD/ct)", "Net t", "Win"]
    parts.append(_table_header(cols))
    rows = []
    # Show test years + pooled row at the bottom (if present)
    wf = wf_summary.copy()
    pooled = wf[wf["test_year"].astype(str) == "pooled"].copy()
    wf = wf[wf["test_year"].astype(str) != "pooled"].copy()
    wf = wf.sort_values("test_year")
    wf = pd.concat([wf, pooled], ignore_index=True)
    for _, row in wf.iterrows():
        rows.append(
            [
                _latex_escape(str(row.get("test_year", ""))),
                _latex_escape(str(row.get("regime", ""))),
                _latex_escape(str(row.get("direction", ""))),
                _fmt_int(row.get("entry_dte", "")),
                _fmt_int(row.get("exit_dte", "")),
                _fmt_int(row.get("n", "")),
                _fmt_money(row.get("net_mean_usd", float("nan"))),
                _fmt_float(row.get("net_t", float("nan")), digits=2),
                _fmt_float(row.get("net_win_rate", float("nan")), digits=2),
            ]
        )
    parts.append(_render_rows(rows))
    parts.append(_table_footer())
    parts.append(r"\normalsize")

    # Table: Robustness scenario comparison (pooled)
    parts.append(r"\section*{Fixed-rule robustness (pooled)}")
    parts.append(r"\footnotesize")
    cols = ["Scenario", "N", "Net mean (USD/ct)", "Net t", "Win"]
    parts.append(_table_header(cols))
    rows = []
    for _, row in robust_summary.iterrows():
        rows.append(
            [
                _latex_escape(str(row.get("scenario", ""))),
                _fmt_int(row.get("n", "")),
                _fmt_money(row.get("net_mean_usd", float("nan"))),
                _fmt_float(row.get("net_t", float("nan")), digits=2),
                _fmt_float(row.get("net_win_rate", float("nan")), digits=2),
            ]
        )
    parts.append(_render_rows(rows))
    parts.append(_table_footer())
    parts.append(r"\normalsize")

    # Table: Baseline fixed-rule by expiry year
    parts.append(r"\section*{Baseline fixed-rule performance by expiry year}")
    parts.append(r"\footnotesize")
    cols = ["Expiry yr", "N", "Net mean (USD/ct)", "Net t", "Win"]
    parts.append(_table_header(cols))
    rows = []
    base = baseline_by_year.copy()
    pooled = base[base["expiry_year"].astype(str) == "pooled"].copy()
    base = base[base["expiry_year"].astype(str) != "pooled"].copy()
    base = base.sort_values("expiry_year")
    base = pd.concat([base, pooled], ignore_index=True)
    for _, row in base.iterrows():
        rows.append(
            [
                _latex_escape(str(row.get("expiry_year", ""))),
                _fmt_int(row.get("n", "")),
                _fmt_money(row.get("net_mean_usd", float("nan"))),
                _fmt_float(row.get("net_t", float("nan")), digits=2),
                _fmt_float(row.get("net_win_rate", float("nan")), digits=2),
            ]
        )
    parts.append(_render_rows(rows))
    parts.append(_table_footer())
    parts.append(r"\normalsize")

    return "\n\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build LaTeX tables for the pre-expiry DTE report.")
    ap.add_argument("--output-root", default="output", help="Repo-level output folder")
    ap.add_argument("--out-tex", default="reports/pre_expiry_dte_report_tables.tex", help="Output .tex path")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    tex = build_tables(output_root=output_root)
    out_tex.write_text(tex, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

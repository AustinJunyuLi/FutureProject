"""Event studies for DTE-anchored spread dynamics.

This module is intentionally "pure" (data in -> stats out), with plotting helpers
that save figures to disk. Generated outputs should live under the repo-level
`output/` folder (gitignored), not under `src/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .expiry_seasonality import align_by_dte


DEFAULT_REGIMES: tuple[str, ...] = ("all", "s2_pos", "s2_neg")


@dataclass(frozen=True)
class EntryRegimeSpec:
    """Defines how to split cycles into regimes at the entry date."""

    entry_dte: int
    s2_col: str = "s2"
    dte_col: str = "dte_bdays"
    cycle_col: str = "near_contract"


def entry_day_s2_sign_by_cycle(
    panel: pd.DataFrame,
    *,
    spec: EntryRegimeSpec,
) -> pd.Series:
    """Return per-cycle S2 sign (True for >=0, False for <0) at entry DTE.

    Cycles without an S2 value at the entry DTE are omitted.
    """
    if spec.cycle_col not in panel.columns or spec.dte_col not in panel.columns:
        raise ValueError("panel missing cycle or DTE columns")
    if spec.s2_col not in panel.columns:
        raise ValueError(f"panel missing {spec.s2_col} column")

    df = panel[[spec.cycle_col, spec.dte_col, spec.s2_col]].copy()
    df = df.dropna(subset=[spec.cycle_col, spec.dte_col])
    df[spec.dte_col] = pd.to_numeric(df[spec.dte_col], errors="coerce")
    df = df.dropna(subset=[spec.dte_col])
    df[spec.dte_col] = df[spec.dte_col].astype(int)

    df = df[df[spec.dte_col] == int(spec.entry_dte)].copy()
    if df.empty:
        return pd.Series(dtype=bool)

    s2_entry = df.groupby(spec.cycle_col, sort=False)[spec.s2_col].mean()
    s2_entry = s2_entry.dropna()
    return (s2_entry >= 0.0).astype(bool)


def cycles_for_regime(
    cycles: Iterable[str],
    *,
    regime: str,
    entry_s2_sign: pd.Series,
) -> list[str]:
    """Filter cycles for a regime using an entry-day S2 sign mapping."""
    regime_norm = regime.strip().lower()
    cycle_list = [str(c) for c in cycles]
    if regime_norm == "all":
        return cycle_list
    if regime_norm not in ("s2_pos", "s2_neg"):
        raise ValueError(f"Unknown regime: {regime}")

    want_pos = regime_norm == "s2_pos"
    keep = entry_s2_sign[entry_s2_sign == want_pos].index.astype(str)
    keep_set = set(keep.tolist())
    return [c for c in cycle_list if c in keep_set]


def _mean_iqr_count(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    frames = []
    for label, s in series_map.items():
        if s is None or s.empty:
            continue
        ss = s.copy()
        ss.name = label
        frames.append(ss)
    if not frames:
        raise ValueError("All series are empty")

    wide = pd.concat(frames, axis=1).sort_index()
    if not wide.index.is_unique:
        wide = wide.groupby(level=0).mean()
    out = pd.DataFrame(
        {
            "mean": wide.mean(axis=1),
            "p25": wide.quantile(0.25, axis=1),
            "p75": wide.quantile(0.75, axis=1),
            "n": wide.count(axis=1),
        }
    )
    out.index.name = "DTE"
    return out


def dte_drift_by_regime(
    panel: pd.DataFrame,
    *,
    dte_max: int,
    value_col: str = "ds1",
    dte_col: str = "dte_bdays",
    cycle_col: str = "near_contract",
    regimes: Iterable[str] = DEFAULT_REGIMES,
    regime_spec: Optional[EntryRegimeSpec] = None,
) -> dict[str, pd.DataFrame]:
    """Compute mean/IQR ΔS1 by DTE, split by entry-day S2 regime."""
    if value_col not in panel.columns:
        raise ValueError(f"panel missing {value_col} column")

    df = panel.dropna(subset=[value_col, dte_col, cycle_col]).copy()
    df = df.rename(columns={value_col: "value"})
    df[cycle_col] = df[cycle_col].astype(str)

    all_cycles = df[cycle_col].unique().tolist()
    entry_spec = regime_spec or EntryRegimeSpec(entry_dte=20, s2_col="s2", dte_col=dte_col, cycle_col=cycle_col)
    entry_s2_sign = entry_day_s2_sign_by_cycle(panel, spec=entry_spec)

    out: dict[str, pd.DataFrame] = {}
    for regime in regimes:
        keep_cycles = cycles_for_regime(all_cycles, regime=regime, entry_s2_sign=entry_s2_sign)
        sub = df[df[cycle_col].isin(keep_cycles)].copy()
        series_map = align_by_dte(sub, dte_col=dte_col, group_col=cycle_col, dte_max=int(dte_max))
        series_map = {k: s.dropna() for k, s in series_map.items() if s is not None and not s.dropna().empty}
        if not series_map:
            out[regime] = pd.DataFrame(columns=["mean", "p25", "p75", "n"])
            continue
        out[regime] = _mean_iqr_count(series_map)
    return out


def cumulative_drift_by_regime(
    panel: pd.DataFrame,
    *,
    dte_max: int,
    baseline_dte: int,
    level_col: str = "s1",
    dte_col: str = "dte_bdays",
    cycle_col: str = "near_contract",
    regimes: Iterable[str] = DEFAULT_REGIMES,
    regime_spec: Optional[EntryRegimeSpec] = None,
) -> dict[str, pd.DataFrame]:
    """Compute mean/IQR of (S1 - S1@baseline_dte) by DTE, split by entry-day S2 regime."""
    if level_col not in panel.columns:
        raise ValueError(f"panel missing {level_col} column")

    df = panel.dropna(subset=[level_col, dte_col, cycle_col]).copy()
    df = df.rename(columns={level_col: "value"})
    df[cycle_col] = df[cycle_col].astype(str)

    all_cycles = df[cycle_col].unique().tolist()
    entry_spec = regime_spec or EntryRegimeSpec(entry_dte=20, s2_col="s2", dte_col=dte_col, cycle_col=cycle_col)
    entry_s2_sign = entry_day_s2_sign_by_cycle(panel, spec=entry_spec)

    out: dict[str, pd.DataFrame] = {}
    for regime in regimes:
        keep_cycles = cycles_for_regime(all_cycles, regime=regime, entry_s2_sign=entry_s2_sign)
        sub = df[df[cycle_col].isin(keep_cycles)].copy()
        series_map = align_by_dte(sub, dte_col=dte_col, group_col=cycle_col, dte_max=int(dte_max))

        adjusted: Dict[str, pd.Series] = {}
        for key, s in series_map.items():
            if s is None or s.empty:
                continue
            if int(baseline_dte) not in s.index:
                continue
            adjusted[key] = s - float(s.loc[int(baseline_dte)])

        adjusted = {k: s.dropna() for k, s in adjusted.items() if not s.dropna().empty}
        if not adjusted:
            out[regime] = pd.DataFrame(columns=["mean", "p25", "p75", "n"])
            continue
        out[regime] = _mean_iqr_count(adjusted)
    return out


def plot_regime_frames(
    frames: dict[str, pd.DataFrame],
    *,
    title_prefix: str,
    ylabel: str,
    output_dir: Path,
    filename_prefix: str,
) -> None:
    """Plot mean/IQR curves for each regime from precomputed frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for regime, df in frames.items():
        if df.empty:
            continue
        # Reconstruct a series_map so we can reuse the existing plot helper.
        # Plot helper expects per-cycle series; we don't have them here, so we
        # just plot the aggregated statistics directly.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = df.index.to_numpy(dtype=int)
        ax.plot(x, df["mean"].to_numpy(dtype=float), color="#1f77b4", linewidth=2.0, label="Mean")
        ax.fill_between(
            x,
            df["p25"].to_numpy(dtype=float),
            df["p75"].to_numpy(dtype=float),
            color="#1f77b4",
            alpha=0.15,
            label="IQR (25-75%)",
        )
        ax.set_title(f"{title_prefix} ({regime})")
        ax.set_xlabel("Business Days to Expiry (DTE)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.invert_xaxis()
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_{regime}.png", dpi=150)
        plt.close(fig)

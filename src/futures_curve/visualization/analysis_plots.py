"""Stage 3 analysis visualizations with academic publication-quality styling."""

import os
from pathlib import Path
from typing import Optional

# Prevent MKL/OpenMP shared-memory usage in restricted environments
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from .styles import (
    COLORS, FIGURE_DEFAULTS, FONT_SETTINGS, PLOT_SETTINGS,
    LAYOUT_SETTINGS, ACADEMIC_PALETTE, apply_style, get_heatmap_text_color
)


class AnalysisVisualizer:
    """Generate Stage 3 analysis visualizations with academic styling."""

    def __init__(self, output_dir: Path | str):
        """Initialize with output directory for figures."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_seasonality_heatmap(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create monthly seasonality heatmap showing win rate and mean return."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Prepare data for heatmaps
        months = df["month_name"].tolist()

        # Win rate heatmap (single row)
        win = df["win_rate"].to_numpy(dtype=float)[None, :]
        vmin_win, vmax_win = 30, 90
        im0 = axes[0].imshow(win, aspect="auto", cmap="RdYlGn", vmin=vmin_win, vmax=vmax_win)
        axes[0].set_xticks(np.arange(len(months)))
        axes[0].set_xticklabels(months, rotation=0, fontsize=FONT_SETTINGS["tick_size"])
        axes[0].set_yticks([0])
        axes[0].set_yticklabels(["Win Rate"], fontsize=FONT_SETTINGS["label_size"])

        # Adaptive text colors for win rate
        for j, val in enumerate(win[0]):
            text_color = get_heatmap_text_color(val, vmin_win, vmax_win, "RdYlGn")
            axes[0].text(j, 0, f"{val:.1f}%", ha="center", va="center",
                        fontsize=FONT_SETTINGS["heatmap_annot_size"], fontweight="bold",
                        color=text_color)

        cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.8)
        cbar0.set_label("Win Rate (%)", fontsize=FONT_SETTINGS["label_size"])
        cbar0.ax.tick_params(labelsize=FONT_SETTINGS["tick_size"])
        axes[0].set_title(
            f"{symbol} EOM Strategy Win Rate by Month",
            fontsize=FONT_SETTINGS["title_size"],
            fontweight="bold",
            pad=12,
        )

        # Mean return heatmap (single row; percent)
        mean_ret = (df["mean_return"].to_numpy(dtype=float) * 100.0)[None, :]
        vmin = float(np.nanmin(mean_ret))
        vmax = float(np.nanmax(mean_ret))
        if vmin < 0.0 < vmax:
            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
            im1 = axes[1].imshow(mean_ret, aspect="auto", cmap="RdYlGn", norm=norm)
        else:
            im1 = axes[1].imshow(mean_ret, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

        axes[1].set_xticks(np.arange(len(months)))
        axes[1].set_xticklabels(months, rotation=0, fontsize=FONT_SETTINGS["tick_size"])
        axes[1].set_yticks([0])
        axes[1].set_yticklabels(["Mean Return"], fontsize=FONT_SETTINGS["label_size"])

        # Adaptive text colors for mean return
        for j, val in enumerate(mean_ret[0]):
            text_color = get_heatmap_text_color(val, vmin, vmax, "RdYlGn")
            axes[1].text(j, 0, f"{val:.2f}%", ha="center", va="center",
                        fontsize=FONT_SETTINGS["heatmap_annot_size"], fontweight="bold",
                        color=text_color)

        cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.8)
        cbar1.set_label("Mean Return (%)", fontsize=FONT_SETTINGS["label_size"])
        cbar1.ax.tick_params(labelsize=FONT_SETTINGS["tick_size"])
        axes[1].set_title(
            f"{symbol} EOM Mean Spread Return by Month",
            fontsize=FONT_SETTINGS["title_size"],
            fontweight="bold",
            pad=12,
        )

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_seasonality_heatmap.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_dte_profile(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create DTE profile showing mean spread and contango percentage."""
        fig, ax1 = plt.subplots(figsize=PLOT_SETTINGS["line"]["figsize"])

        x = np.arange(len(df))
        width = 0.6

        # Bar chart for contango percentage
        bars = ax1.bar(
            x,
            df["pct_contango"],
            width,
            color=ACADEMIC_PALETTE["positive"],
            alpha=0.7,
            edgecolor=ACADEMIC_PALETTE["primary"],
            linewidth=LAYOUT_SETTINGS["bar_edge_width"],
            label="% Contango",
        )
        ax1.axhline(y=50, color=ACADEMIC_PALETTE["neutral"], linestyle="--",
                   linewidth=1.5, alpha=0.8, label="50% Reference")
        ax1.set_ylabel("Contango Percentage (%)", fontsize=FONT_SETTINGS["label_size"],
                      color=ACADEMIC_PALETTE["positive"])
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='y', labelcolor=ACADEMIC_PALETTE["positive"])

        # Line chart for mean spread on secondary axis
        ax2 = ax1.twinx()
        line = ax2.plot(
            x,
            df["mean"] * 100,  # Convert to percentage points
            color=ACADEMIC_PALETTE["primary"],
            linewidth=LAYOUT_SETTINGS["line_width"],
            marker=PLOT_SETTINGS["line"]["marker"],
            markersize=LAYOUT_SETTINGS["marker_size"],
            label="Mean Spread (%)",
        )
        ax2.axhline(y=0, color=ACADEMIC_PALETTE["negative"], linestyle="-",
                   linewidth=1, alpha=0.5)
        ax2.set_ylabel(
            "Mean Spread (%)", fontsize=FONT_SETTINGS["label_size"],
            color=ACADEMIC_PALETTE["primary"]
        )
        ax2.tick_params(axis='y', labelcolor=ACADEMIC_PALETTE["primary"])

        # X-axis labels
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["dte_bin"], fontsize=FONT_SETTINGS["tick_size"])
        ax1.set_xlabel("Days to Expiry", fontsize=FONT_SETTINGS["label_size"])

        # Title and legend
        ax1.set_title(
            f"{symbol} Spread Characteristics by DTE",
            fontsize=FONT_SETTINGS["title_size"],
            fontweight="bold",
            pad=12,
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                  fontsize=FONT_SETTINGS["legend_size"], framealpha=0.9)

        ax1.grid(True, alpha=LAYOUT_SETTINGS["grid_alpha"],
                linestyle=LAYOUT_SETTINGS["grid_linestyle"])
        ax1.spines["top"].set_visible(False)

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_dte_profile.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_roll_event_study(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create roll event study showing mean spread around roll events with CI bands."""
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS["line"]["figsize"])

        rel_col = "rel_bday" if "rel_bday" in df.columns else "rel_day"

        # Filter to reasonable range around event
        df_plot = df[(df[rel_col] >= -10) & (df[rel_col] <= 10)].copy()

        # Plot mean line
        ax.plot(
            df_plot[rel_col],
            df_plot["mean"] * 100,  # Convert to percentage
            color=ACADEMIC_PALETTE["primary"],
            linewidth=LAYOUT_SETTINGS["line_width"],
            marker="o",
            markersize=LAYOUT_SETTINGS["marker_size"],
            label="Mean Spread",
        )

        # Plot confidence interval bands
        ax.fill_between(
            df_plot[rel_col],
            df_plot["ci_lower"] * 100,
            df_plot["ci_upper"] * 100,
            color=ACADEMIC_PALETTE["ci_band"],
            alpha=0.25,
            label="95% CI",
        )

        # Mark the roll event day
        ax.axvline(x=0, color=ACADEMIC_PALETTE["negative"], linestyle="--",
                  linewidth=2, label="Roll Day")
        ax.axhline(y=0, color=ACADEMIC_PALETTE["neutral"], linestyle="-",
                  linewidth=1, alpha=0.5)

        apply_style(
            ax,
            title=f"{symbol} Spread Behavior Around Roll Events",
            xlabel="Business Days Relative to Roll",
            ylabel="Mean Spread (%)",
        )
        ax.legend(loc="upper right", fontsize=FONT_SETTINGS["legend_size"],
                 framealpha=0.9)

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_roll_event_study.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_eom_returns_distribution(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create histogram of EOM returns colored by sign."""
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS["histogram"]["figsize"])

        returns = df["spread_return"] * 100  # Convert to percentage

        # Separate positive and negative returns
        pos_returns = returns[returns >= 0]
        neg_returns = returns[returns < 0]

        bins = np.linspace(returns.min(), returns.max(), PLOT_SETTINGS["histogram"]["bins"])

        # Plot positive returns
        ax.hist(
            pos_returns,
            bins=bins,
            color=ACADEMIC_PALETTE["positive"],
            alpha=PLOT_SETTINGS["histogram"]["alpha"],
            edgecolor=PLOT_SETTINGS["histogram"]["edgecolor"],
            linewidth=PLOT_SETTINGS["histogram"]["linewidth"],
            label=f"Gains (n={len(pos_returns)})",
        )

        # Plot negative returns
        ax.hist(
            neg_returns,
            bins=bins,
            color=ACADEMIC_PALETTE["negative"],
            alpha=PLOT_SETTINGS["histogram"]["alpha"],
            edgecolor=PLOT_SETTINGS["histogram"]["edgecolor"],
            linewidth=PLOT_SETTINGS["histogram"]["linewidth"],
            label=f"Losses (n={len(neg_returns)})",
        )

        # Add statistics
        mean_ret = returns.mean()
        median_ret = returns.median()
        ax.axvline(
            x=mean_ret,
            color=ACADEMIC_PALETTE["primary"],
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_ret:.2f}%",
        )
        ax.axvline(
            x=median_ret,
            color=ACADEMIC_PALETTE["accent1"],
            linestyle=":",
            linewidth=2,
            label=f"Median: {median_ret:.2f}%",
        )

        apply_style(
            ax,
            title=f"{symbol} End-of-Month Spread Return Distribution",
            xlabel="Spread Return (%)",
            ylabel="Frequency",
        )
        ax.legend(loc="upper right", fontsize=FONT_SETTINGS["legend_size"],
                 framealpha=0.9)

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_eom_returns_dist.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    # ========== NEW PLOTS ==========

    def plot_contango_backwardation_timeline(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create timeline showing contango/backwardation state over time."""
        fig, ax = plt.subplots(figsize=(14, 5))

        # Ensure we have date column
        if "trade_date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["trade_date"])
        elif "date" not in df.columns:
            plt.close(fig)
            raise ValueError("DataFrame needs 'trade_date' or 'date' column")

        # Use S1_pct if available, otherwise S1_raw
        spread_col = "S1_pct" if "S1_pct" in df.columns else "S1_raw"
        if spread_col not in df.columns:
            plt.close(fig)
            raise ValueError(f"DataFrame needs '{spread_col}' column")

        # Daily aggregation
        daily = df.groupby("date")[spread_col].mean().reset_index()
        daily = daily.sort_values("date")

        dates = daily["date"]
        spreads = daily[spread_col] * 100 if spread_col == "S1_pct" else daily[spread_col]

        # Color by sign
        colors = [ACADEMIC_PALETTE["positive"] if s >= 0 else ACADEMIC_PALETTE["negative"]
                  for s in spreads]

        ax.bar(dates, spreads, color=colors, width=1.5, alpha=0.8)
        ax.axhline(y=0, color=ACADEMIC_PALETTE["neutral"], linestyle="-", linewidth=1.5)

        # Add rolling average
        if len(spreads) > 20:
            rolling_mean = spreads.rolling(window=20, min_periods=1).mean()
            ax.plot(dates, rolling_mean, color=ACADEMIC_PALETTE["primary"],
                   linewidth=2, label="20-day MA", alpha=0.9)
            ax.legend(loc="upper right", fontsize=FONT_SETTINGS["legend_size"])

        apply_style(
            ax,
            title=f"{symbol} Contango/Backwardation Timeline",
            xlabel="Date",
            ylabel="Spread (%)" if spread_col == "S1_pct" else "Spread (Price Units)",
        )

        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        fig.autofmt_xdate()

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_contango_timeline.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_volume_share_evolution(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create chart showing F2 volume share evolution over the front-month lifecycle.

        We aggregate volumes across the US session buckets (1--7) per trade date:

            share_day = sum(F2_volume) / (sum(F1_volume) + sum(F2_volume))

        Then we summarize by F1 business-day DTE so the figure reflects the
        typical roll progression (an S-curve) rather than a visually dense
        multi-year time series.
        """
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS["line"]["figsize"])

        df = df.copy()

        # Date column
        if "trade_date" in df.columns:
            df["date"] = pd.to_datetime(df["trade_date"])
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            plt.close(fig)
            raise ValueError("DataFrame needs 'trade_date' or 'date' column")

        # Preferred path: compute daily share from summed volumes (robust).
        if "F1_volume" in df.columns and "F2_volume" in df.columns:
            df["F1_volume"] = pd.to_numeric(df["F1_volume"], errors="coerce")
            df["F2_volume"] = pd.to_numeric(df["F2_volume"], errors="coerce")

            # Use US session buckets (1-7) when available to match the daily proxy convention.
            if "bucket" in df.columns:
                df = df[df["bucket"].between(1, 7)].copy()

            daily = (
                df.groupby("date", as_index=False)
                .agg(F1_volume=("F1_volume", "sum"), F2_volume=("F2_volume", "sum"))
                .sort_values("date")
            )
            total = daily["F1_volume"] + daily["F2_volume"]
            daily["vol_share"] = daily["F2_volume"] / total.replace(0, np.nan)
            # Attach daily F1 DTE (trade-date based, bucket-invariant).
            if "F1_dte_bdays" in df.columns:
                dte_map = (
                    df.groupby("date", as_index=False)["F1_dte_bdays"]
                    .first()
                    .rename(columns={"F1_dte_bdays": "dte"})
                )
                daily = daily.merge(dte_map, on="date", how="left")
        else:
            # Fallback: use a provided share column; if total volume exists, compute
            # a volume-weighted average rather than a simple mean.
            vol_share_col = None
            for col in ["vol_share", "volume_share", "f2_volume_share"]:
                if col in df.columns:
                    vol_share_col = col
                    break
            if vol_share_col is None:
                plt.close(fig)
                raise ValueError("DataFrame needs F1/F2 volumes or a volume share column")

            if "total_volume" in df.columns:
                df["total_volume"] = pd.to_numeric(df["total_volume"], errors="coerce")
                df["__w"] = df["total_volume"].where(df["total_volume"] > 0)
                daily = (
                    df.groupby("date", as_index=False)
                    .apply(lambda g: pd.Series({
                        "vol_share": (g[vol_share_col] * g["__w"]).sum() / g["__w"].sum()
                        if g["__w"].notna().any() else np.nan
                    }))
                    .reset_index(drop=True)
                )
            else:
                daily = df.groupby("date", as_index=False)[vol_share_col].mean().rename(columns={vol_share_col: "vol_share"})

            daily = daily.sort_values("date")
            # Attach daily F1 DTE if available.
            if "F1_dte_bdays" in df.columns:
                dte_map = (
                    df.groupby("date", as_index=False)["F1_dte_bdays"]
                    .first()
                    .rename(columns={"F1_dte_bdays": "dte"})
                )
                daily = daily.merge(dte_map, on="date", how="left")

        # Build the lifecycle profile: share vs DTE.
        if "dte" not in daily.columns:
            plt.close(fig)
            raise ValueError("DataFrame needs F1_dte_bdays to plot lifecycle volume-share profile")

        prof = daily.copy()
        prof["dte"] = pd.to_numeric(prof["dte"], errors="coerce")
        prof = prof.dropna(subset=["dte", "vol_share"]).copy()
        if prof.empty:
            plt.close(fig)
            raise ValueError("No valid volume share / DTE data to plot")

        prof["dte"] = prof["dte"].round().astype(int)
        stats = (
            prof.groupby("dte")["vol_share"]
            .agg(
                count="count",
                mean="mean",
                p25=lambda s: s.quantile(0.25),
                p75=lambda s: s.quantile(0.75),
            )
            .reset_index()
            .sort_values("dte")
        )

        # Keep only DTE values with sufficient sample support to avoid single-observation noise.
        stats = stats[stats["count"] >= 30].copy()
        if stats.empty:
            plt.close(fig)
            raise ValueError("Insufficient sample size to build DTE profile (need >=30 observations per DTE)")

        dte = stats["dte"].to_numpy()
        mean_pct = stats["mean"].to_numpy() * 100
        p25_pct = stats["p25"].to_numpy() * 100
        p75_pct = stats["p75"].to_numpy() * 100

        ax.plot(
            dte,
            mean_pct,
            color=ACADEMIC_PALETTE["primary"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="Mean (US-session, daily)",
        )
        ax.fill_between(
            dte,
            p25_pct,
            p75_pct,
            color=ACADEMIC_PALETTE["ci_band"],
            alpha=0.25,
            label="IQR (25–75%)",
        )

        # Reverse x-axis so the roll progression reads left->right (higher DTE -> lower DTE).
        ax.invert_xaxis()

        # Reference lines for roll thresholds
        ax.axhline(y=25, color=ACADEMIC_PALETTE["neutral"], linestyle="--",
                  linewidth=1.5, alpha=0.7, label="Roll Start (25%)")
        ax.axhline(y=50, color=ACADEMIC_PALETTE["secondary"], linestyle="-.",
                  linewidth=1.5, alpha=0.7, label="Roll Peak (50%)")
        ax.axhline(y=75, color=ACADEMIC_PALETTE["neutral"], linestyle="--",
                  linewidth=1.5, alpha=0.7, label="Roll End (75%)")

        ax.set_ylim(0, 100)

        apply_style(
            ax,
            title=f"{symbol} F2 Volume Share vs F1 DTE (roll progression)",
            xlabel="F1 DTE (business days to expiry)",
            ylabel="F2 Volume Share (%)",
        )
        ax.legend(loc="lower left", fontsize=FONT_SETTINGS["legend_size"],
                 framealpha=0.9, ncol=2)

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_volume_share.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_bucket_seasonality_grid(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create grid showing spread patterns by bucket and month."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Need bucket and month columns
        if "bucket" not in df.columns:
            plt.close(fig)
            raise ValueError("DataFrame needs 'bucket' column")

        # Get date/month info
        if "trade_date" in df.columns:
            df = df.copy()
            df["month"] = pd.to_datetime(df["trade_date"]).dt.month
        elif "month" not in df.columns:
            plt.close(fig)
            raise ValueError("DataFrame needs date info")

        # Use S1_pct if available
        spread_col = "S1_pct" if "S1_pct" in df.columns else "S1_raw"
        if spread_col not in df.columns:
            plt.close(fig)
            raise ValueError(f"DataFrame needs '{spread_col}' column")

        # Create pivot table: buckets x months
        pivot = df.pivot_table(
            values=spread_col,
            index="bucket",
            columns="month",
            aggfunc="mean"
        ) * (100 if spread_col == "S1_pct" else 1)

        # Rename columns to month names
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.columns = [month_names[m-1] for m in pivot.columns]

        values = pivot.to_numpy(dtype=float)
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))

        if vmin < 0.0 < vmax:
            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
            im = ax.imshow(values, aspect="auto", cmap="RdYlGn", norm=norm)
        else:
            im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), fontsize=FONT_SETTINGS["tick_size"])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"Bucket {b}" for b in pivot.index],
                         fontsize=FONT_SETTINGS["tick_size"])

        # Annotate cells with adaptive text color
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if not np.isnan(val):
                    text_color = get_heatmap_text_color(val, vmin, vmax, "RdYlGn")
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=FONT_SETTINGS["heatmap_annot_size"],
                           color=text_color, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Mean Spread (%)" if spread_col == "S1_pct" else "Mean Spread",
                      fontsize=FONT_SETTINGS["label_size"])
        cbar.ax.tick_params(labelsize=FONT_SETTINGS["tick_size"])

        ax.set_title(
            f"{symbol} Spread by Bucket and Month",
            fontsize=FONT_SETTINGS["title_size"],
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Month", fontsize=FONT_SETTINGS["label_size"])
        ax.set_ylabel("Trading Bucket", fontsize=FONT_SETTINGS["label_size"])

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_bucket_seasonality.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_term_structure_snapshot(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create term structure snapshot showing price curve across contracts."""
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS["line"]["figsize"])

        # Look for price columns F1_price, F2_price, ... F12_price
        price_cols = [col for col in df.columns if col.startswith("F") and col.endswith("_price")]
        price_cols = sorted(price_cols, key=lambda x: int(x[1:].split("_")[0]))

        if len(price_cols) < 2:
            plt.close(fig)
            raise ValueError("DataFrame needs F1, F2, ... price columns")

        df = df.copy()

        # Prefer a chronologically latest row with usable prices (avoid blank plots).
        if "ts_end_utc" in df.columns:
            df["_order_ts"] = pd.to_datetime(df["ts_end_utc"])
        elif "trade_date" in df.columns:
            df["_order_ts"] = pd.to_datetime(df["trade_date"])
        else:
            df["_order_ts"] = np.arange(len(df))

        df = df.sort_values("_order_ts").reset_index(drop=True)

        # Choose a snapshot with a *meaningful* curve depth (avoid end-of-sample truncation
        # where only 1-2 contracts remain in the dataset).
        non_na_counts = df[price_cols].notna().sum(axis=1)

        # Prefer the deepest available strip; within that, prefer a snapshot where the
        # *near* contracts have non-trivial volume so F1/F2 aren't just 1-lot marks.
        snapshot = None
        max_points = int(non_na_counts.max()) if len(non_na_counts) else 0
        for min_points in range(min(12, max_points), 1, -1):
            mask = non_na_counts >= min_points
            if not mask.any():
                continue

            candidates = df.loc[mask].copy()

            # If volume columns exist, require a minimum near-month activity threshold when possible.
            if "F1_volume" in candidates.columns and "F2_volume" in candidates.columns:
                candidates["_near_vol"] = candidates["F1_volume"].fillna(0.0) + candidates["F2_volume"].fillna(0.0)
                near_vol_min = 20.0
                liquid = candidates[candidates["_near_vol"] >= near_vol_min]
                if not liquid.empty:
                    candidates = liquid

            # Pick the most recent candidate within this depth class.
            snapshot = candidates.sort_values("_order_ts").iloc[-1]
            break

        if snapshot is None:
            plt.close(fig)
            raise ValueError("No term-structure snapshot with at least two valid prices.")

        latest = snapshot

        # Extract prices and labels from non-null columns
        valid_price_cols = [col for col in price_cols if pd.notna(latest.get(col, np.nan))]
        prices = [latest[col] for col in valid_price_cols]

        # Build labels like "F1\nHGF24" when contract codes are available.
        contracts = []
        for col in valid_price_cols:
            f = col.replace("_price", "")  # e.g. "F1"
            contract_code = latest.get(f"{f}_contract") if f"{f}_contract" in latest.index else None
            if isinstance(contract_code, str) and contract_code:
                contracts.append(f"{f}\n{contract_code}")
            else:
                contracts.append(f)

        # Label for title
        if "trade_date" in df.columns:
            latest_date = pd.to_datetime(latest.get("trade_date")).date()
            if "bucket" in df.columns and pd.notna(latest.get("bucket")):
                latest_date = f"{latest_date} (bucket {int(latest.get('bucket'))})"
        else:
            latest_date = "Latest"

        # Plot term structure
        x = np.arange(len(contracts))
        ax.plot(
            x,
            prices,
            color=ACADEMIC_PALETTE["primary"],
            linewidth=LAYOUT_SETTINGS["line_width"],
            label="Price",
        )

        # Liquidity cues: marker size/alpha based on bucket volume when available.
        volumes = []
        for col in valid_price_cols:
            f = col.replace("_price", "")
            v = latest.get(f"{f}_volume") if f"{f}_volume" in latest.index else None
            try:
                volumes.append(float(v) if v is not None and not pd.isna(v) else 0.0)
            except Exception:
                volumes.append(0.0)

        if volumes:
            v = np.array(volumes, dtype="float64")
            v_log = np.log10(v + 1.0)
            v_log_max = float(v_log.max()) if float(v_log.max()) > 0 else 1.0
            # Marker size and alpha are scaled so tiny-volume points are visibly de-emphasized.
            sizes = (LAYOUT_SETTINGS["marker_size"] * 6.0) * (0.3 + 1.7 * (v_log / v_log_max))
            alphas = 0.25 + 0.75 * (v_log / v_log_max)

            for xi, yi, si, ai in zip(x, prices, sizes, alphas):
                ax.scatter(
                    [xi],
                    [yi],
                    s=float(si) ** 2,
                    facecolors="white",
                    edgecolors=ACADEMIC_PALETTE["primary"],
                    linewidths=2.0,
                    alpha=float(ai),
                    zorder=3,
                )

        # Add price labels
        for i, (contract, price) in enumerate(zip(contracts, prices)):
            ax.annotate(f"${price:.4f}", (i, price), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=FONT_SETTINGS["annotation_size"],
                       color=ACADEMIC_PALETTE["primary"])

        ax.set_xticks(x)
        ax.set_xticklabels(contracts, fontsize=FONT_SETTINGS["tick_size"])

        # Determine contango/backwardation from the front spread (F2 - F1) when available.
        if len(prices) >= 2:
            f1_price = float(prices[0])
            f2_price = float(prices[1])
            if f2_price >= f1_price:
                shape = "Contango"
                shape_color = ACADEMIC_PALETTE["positive"]
            else:
                shape = "Backwardation"
                shape_color = ACADEMIC_PALETTE["negative"]

            ax.text(0.98, 0.95, shape, transform=ax.transAxes, ha="right", va="top",
                   fontsize=FONT_SETTINGS["subtitle_size"], fontweight="bold",
                   color=shape_color, bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor="white", edgecolor=shape_color,
                                                alpha=0.9))

        apply_style(
            ax,
            title=f"{symbol} Term Structure Snapshot ({latest_date})",
            xlabel="Contract",
            ylabel="Price",
        )

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_term_structure.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def generate_all(self, symbol: str, tables_dir: Path | str,
                    data_dir: Optional[Path | str] = None) -> dict[str, Path]:
        """Generate all Stage 3 analysis plots from saved parquet tables."""
        tables_dir = Path(tables_dir)
        outputs = {}

        def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
            return (not df.empty) and all(c in df.columns for c in cols)

        # Seasonality heatmap
        seasonal_path = tables_dir / f"{symbol}_seasonal_summary.parquet"
        if seasonal_path.exists():
            df = pd.read_parquet(seasonal_path)
            if _has_cols(df, ["month_name", "win_rate", "mean_return"]):
                outputs["seasonality"] = self.plot_seasonality_heatmap(df, symbol)

        # DTE profile
        dte_path = tables_dir / f"{symbol}_dte_profile.parquet"
        if dte_path.exists():
            df = pd.read_parquet(dte_path)
            if _has_cols(df, ["dte_bin", "pct_contango", "mean"]):
                outputs["dte_profile"] = self.plot_dte_profile(df, symbol)

        # Roll event study
        event_path = tables_dir / f"{symbol}_avg_event_curve.parquet"
        if event_path.exists():
            df = pd.read_parquet(event_path)
            if _has_cols(df, ["mean"]) and ("rel_bday" in df.columns or "rel_day" in df.columns):
                outputs["roll_event"] = self.plot_roll_event_study(df, symbol)

        # EOM returns distribution
        eom_path = tables_dir / f"{symbol}_eom_returns.parquet"
        if eom_path.exists():
            df = pd.read_parquet(eom_path)
            if _has_cols(df, ["spread_return"]):
                outputs["eom_returns"] = self.plot_eom_returns_distribution(df, symbol)

        # NEW: Try to load spread data for new plots
        if data_dir is not None:
            data_dir = Path(data_dir)
            spread_path = data_dir / "spreads" / symbol / "spreads_panel.parquet"
            curve_path = data_dir / "curve" / symbol / "curve_panel.parquet"

            if spread_path.exists():
                try:
                    spread_df = pd.read_parquet(spread_path)
                    if not spread_df.empty:
                        outputs["contango_timeline"] = self.plot_contango_backwardation_timeline(
                            spread_df, symbol
                        )
                        outputs["volume_share"] = self.plot_volume_share_evolution(
                            spread_df, symbol
                        )
                        outputs["bucket_seasonality"] = self.plot_bucket_seasonality_grid(
                            spread_df, symbol
                        )
                except Exception:
                    pass  # Skip if data format doesn't match

            if curve_path.exists():
                try:
                    curve_df = pd.read_parquet(curve_path)
                    if not curve_df.empty:
                        outputs["term_structure"] = self.plot_term_structure_snapshot(
                            curve_df, symbol
                        )
                except Exception:
                    pass  # Skip if data format doesn't match

        return outputs

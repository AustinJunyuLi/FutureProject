"""Stage 4 backtest visualizations with academic publication-quality styling."""

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


class BacktestVisualizer:
    """Generate Stage 4 backtest visualizations with academic styling."""

    def __init__(self, output_dir: Path | str):
        """Initialize with output directory for figures."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_equity_curve(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy: str = "eom",
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create equity curve with drawdown overlay."""
        fig, ax1 = plt.subplots(figsize=PLOT_SETTINGS["equity_curve"]["figsize"])

        # Clean data
        df_clean = df.dropna(subset=["equity", "cumulative_pnl"])
        if df_clean.empty:
            plt.close(fig)
            raise ValueError("No valid equity data to plot")

        dates = pd.to_datetime(df_clean["exit_date"])
        equity = df_clean["equity"]

        # Calculate drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100

        # Plot equity curve
        ax1.plot(
            dates,
            equity,
            color=ACADEMIC_PALETTE["primary"],
            linewidth=LAYOUT_SETTINGS["line_width"],
            label="Equity",
        )
        ax1.fill_between(
            dates,
            100000,  # Starting equity
            equity,
            alpha=0.15,
            color=ACADEMIC_PALETTE["ci_band"],
        )
        ax1.set_ylabel("Equity ($)", fontsize=FONT_SETTINGS["label_size"])
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Plot drawdown on secondary axis
        ax2 = ax1.twinx()
        ax2.fill_between(
            dates,
            drawdown,
            0,
            alpha=0.35,
            color=ACADEMIC_PALETTE["negative"],
            label="Drawdown",
        )
        ax2.set_ylabel(
            "Drawdown (%)", fontsize=FONT_SETTINGS["label_size"],
            color=ACADEMIC_PALETTE["negative"]
        )
        ax2.tick_params(axis='y', labelcolor=ACADEMIC_PALETTE["negative"])
        ax2.set_ylim(drawdown.min() * 1.2, 5)

        ax1.set_title(
            f"{symbol} {strategy.upper()} Strategy Equity Curve",
            fontsize=FONT_SETTINGS["title_size"],
            fontweight="bold",
            pad=12,
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
                  fontsize=FONT_SETTINGS["legend_size"], framealpha=0.9)

        ax1.grid(True, alpha=LAYOUT_SETTINGS["grid_alpha"],
                linestyle=LAYOUT_SETTINGS["grid_linestyle"])
        ax1.spines["top"].set_visible(False)

        fig.autofmt_xdate()
        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_{strategy}_equity.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_strategy_comparison(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create grouped bar chart comparing strategies."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        strategies = df["strategy"].tolist()
        x = np.arange(len(strategies))
        width = PLOT_SETTINGS["bar"]["width"]

        # P&L comparison
        colors = [ACADEMIC_PALETTE["positive"] if p > 0 else ACADEMIC_PALETTE["negative"]
                  for p in df["total_pnl"]]
        bars1 = axes[0].bar(x, df["total_pnl"], width, color=colors,
                           edgecolor=ACADEMIC_PALETTE["primary"],
                           linewidth=LAYOUT_SETTINGS["bar_edge_width"])
        axes[0].axhline(y=0, color=ACADEMIC_PALETTE["neutral"], linestyle="-",
                       linewidth=1.5, alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([s.upper() for s in strategies],
                               fontsize=FONT_SETTINGS["tick_size"])
        apply_style(
            axes[0],
            title=f"{symbol} Total P&L by Strategy",
            xlabel="Strategy",
            ylabel="Total P&L ($)",
        )
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Add value labels on bars
        for bar, val in zip(bars1, df["total_pnl"]):
            axes[0].annotate(
                f"${val:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                ha="center",
                va="bottom" if val > 0 else "top",
                fontsize=FONT_SETTINGS["annotation_size"],
                fontweight="bold",
                color=ACADEMIC_PALETTE["primary"],
            )

        # Sharpe ratio comparison
        colors = [ACADEMIC_PALETTE["positive"] if s > 0 else ACADEMIC_PALETTE["negative"]
                  for s in df["sharpe_ratio"]]
        bars2 = axes[1].bar(x, df["sharpe_ratio"], width, color=colors,
                           edgecolor=ACADEMIC_PALETTE["primary"],
                           linewidth=LAYOUT_SETTINGS["bar_edge_width"])
        axes[1].axhline(y=0, color=ACADEMIC_PALETTE["neutral"], linestyle="-",
                       linewidth=1.5, alpha=0.7)
        axes[1].axhline(y=1, color=ACADEMIC_PALETTE["positive"], linestyle="--",
                       linewidth=1.5, alpha=0.6, label="Sharpe=1")
        axes[1].axhline(y=-1, color=ACADEMIC_PALETTE["negative"], linestyle="--",
                       linewidth=1.5, alpha=0.6, label="Sharpe=-1")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([s.upper() for s in strategies],
                               fontsize=FONT_SETTINGS["tick_size"])
        apply_style(
            axes[1],
            title=f"{symbol} Sharpe Ratio by Strategy",
            xlabel="Strategy",
            ylabel="Sharpe Ratio",
        )
        axes[1].legend(fontsize=FONT_SETTINGS["legend_size"], framealpha=0.9)

        # Add value labels on bars
        for bar, val in zip(bars2, df["sharpe_ratio"]):
            axes[1].annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                ha="center",
                va="bottom" if val > 0 else "top",
                fontsize=FONT_SETTINGS["annotation_size"],
                fontweight="bold",
                color=ACADEMIC_PALETTE["primary"],
            )

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_strategy_comparison.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_trade_pnl_distribution(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy: str = "eom",
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create histogram of trade P&L distribution."""
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS["histogram"]["figsize"])

        # Clean data
        pnl = df["pnl"].dropna()
        if pnl.empty:
            plt.close(fig)
            raise ValueError("No valid P&L data to plot")

        # Separate wins and losses
        wins = pnl[pnl >= 0]
        losses = pnl[pnl < 0]

        bins = np.linspace(pnl.min(), pnl.max(), PLOT_SETTINGS["histogram"]["bins"])

        # Plot wins
        ax.hist(
            wins,
            bins=bins,
            color=ACADEMIC_PALETTE["positive"],
            alpha=PLOT_SETTINGS["histogram"]["alpha"],
            edgecolor=PLOT_SETTINGS["histogram"]["edgecolor"],
            linewidth=PLOT_SETTINGS["histogram"]["linewidth"],
            label=f"Wins (n={len(wins)})",
        )

        # Plot losses
        ax.hist(
            losses,
            bins=bins,
            color=ACADEMIC_PALETTE["negative"],
            alpha=PLOT_SETTINGS["histogram"]["alpha"],
            edgecolor=PLOT_SETTINGS["histogram"]["edgecolor"],
            linewidth=PLOT_SETTINGS["histogram"]["linewidth"],
            label=f"Losses (n={len(losses)})",
        )

        # Add statistics
        mean_pnl = pnl.mean()
        median_pnl = pnl.median()
        ax.axvline(
            x=mean_pnl,
            color=ACADEMIC_PALETTE["primary"],
            linestyle="--",
            linewidth=2,
            label=f"Mean: ${mean_pnl:,.0f}",
        )
        ax.axvline(
            x=median_pnl,
            color=ACADEMIC_PALETTE["accent1"],
            linestyle=":",
            linewidth=2,
            label=f"Median: ${median_pnl:,.0f}",
        )

        apply_style(
            ax,
            title=f"{symbol} {strategy.upper()} Trade P&L Distribution",
            xlabel="Trade P&L ($)",
            ylabel="Frequency",
        )
        ax.legend(loc="upper right", fontsize=FONT_SETTINGS["legend_size"],
                 framealpha=0.9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_{strategy}_pnl_dist.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_monthly_returns_heatmap(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy: str = "eom",
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create year x month heatmap of returns."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Clean and prepare data
        df_clean = df.dropna(subset=["pnl", "exit_date"]).copy()
        if df_clean.empty:
            plt.close(fig)
            raise ValueError("No valid trade data for heatmap")

        df_clean["exit_date"] = pd.to_datetime(df_clean["exit_date"])
        df_clean["year"] = df_clean["exit_date"].dt.year
        df_clean["month"] = df_clean["exit_date"].dt.month

        # Aggregate by year/month
        pivot = df_clean.pivot_table(
            values="pnl", index="year", columns="month", aggfunc="sum", fill_value=0
        )

        # Rename columns to month names
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]

        values = pivot.to_numpy(dtype=float)
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if vmin < 0.0 < vmax:
            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
            im = ax.imshow(values, aspect="auto", cmap="RdYlGn", norm=norm)
        else:
            im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(list(pivot.columns), fontsize=FONT_SETTINGS["tick_size"])
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([str(y) for y in pivot.index.tolist()],
                         fontsize=FONT_SETTINGS["tick_size"])

        # Annotate cells with adaptive text color
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                text_color = get_heatmap_text_color(val, vmin, vmax, "RdYlGn")
                ax.text(j, i, f"${val:.0f}", ha="center", va="center",
                       fontsize=FONT_SETTINGS["heatmap_annot_size"],
                       color=text_color, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Monthly P&L ($)", fontsize=FONT_SETTINGS["label_size"])
        cbar.ax.tick_params(labelsize=FONT_SETTINGS["tick_size"])

        ax.set_title(
            f"{symbol} {strategy.upper()} Monthly P&L Heatmap",
            fontsize=FONT_SETTINGS["title_size"],
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Month", fontsize=FONT_SETTINGS["label_size"])
        ax.set_ylabel("Year", fontsize=FONT_SETTINGS["label_size"])

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_{strategy}_monthly_heatmap.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    # ========== NEW PLOTS ==========

    def plot_cumulative_pnl_by_month(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy: str = "eom",
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create chart showing cumulative P&L broken down by calendar month."""
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS["line"]["figsize"])

        # Clean data
        df_clean = df.dropna(subset=["pnl", "exit_date"]).copy()
        if df_clean.empty:
            plt.close(fig)
            raise ValueError("No valid trade data")

        df_clean["exit_date"] = pd.to_datetime(df_clean["exit_date"])
        df_clean["month"] = df_clean["exit_date"].dt.month

        # Get month names
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # Calculate cumulative P&L per month
        monthly_totals = df_clean.groupby("month")["pnl"].sum()
        months = [month_names[m-1] for m in monthly_totals.index]
        values = monthly_totals.values

        # Calculate cumulative sum
        cumulative = np.cumsum(values)

        x = np.arange(len(months))

        # Bar chart for monthly P&L
        colors = [ACADEMIC_PALETTE["positive"] if v >= 0 else ACADEMIC_PALETTE["negative"]
                  for v in values]
        bars = ax.bar(x, values, color=colors, alpha=0.7,
                     edgecolor=ACADEMIC_PALETTE["primary"],
                     linewidth=LAYOUT_SETTINGS["bar_edge_width"],
                     label="Monthly P&L")

        # Overlay cumulative line
        ax2 = ax.twinx()
        ax2.plot(x, cumulative, color=ACADEMIC_PALETTE["primary"],
                linewidth=LAYOUT_SETTINGS["line_width"],
                marker="o", markersize=LAYOUT_SETTINGS["marker_size"],
                label="Cumulative P&L")
        ax2.axhline(y=0, color=ACADEMIC_PALETTE["neutral"], linestyle="-",
                   linewidth=1, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(months, fontsize=FONT_SETTINGS["tick_size"])
        ax.set_ylabel("Monthly P&L ($)", fontsize=FONT_SETTINGS["label_size"])
        ax2.set_ylabel("Cumulative P&L ($)", fontsize=FONT_SETTINGS["label_size"],
                      color=ACADEMIC_PALETTE["primary"])
        ax2.tick_params(axis='y', labelcolor=ACADEMIC_PALETTE["primary"])

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        ax.set_title(
            f"{symbol} {strategy.upper()} Cumulative P&L by Month",
            fontsize=FONT_SETTINGS["title_size"],
            fontweight="bold",
            pad=12,
        )

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
                 fontsize=FONT_SETTINGS["legend_size"], framealpha=0.9)

        ax.grid(True, alpha=LAYOUT_SETTINGS["grid_alpha"],
               linestyle=LAYOUT_SETTINGS["grid_linestyle"])
        ax.spines["top"].set_visible(False)

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_{strategy}_cumulative_monthly.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_cost_sensitivity(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy: str = "eom",
        base_slippage: float = 1.0,
        base_commission: float = 2.50,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create chart showing how P&L changes with transaction costs."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Clean data
        df_clean = df.dropna(subset=["pnl"]).copy()
        if df_clean.empty:
            plt.close(fig)
            raise ValueError("No valid trade data")

        # Estimate gross P&L (before costs) based on trade count
        n_trades = len(df_clean)
        total_pnl = df_clean["pnl"].sum()

        # Assume 4 fills per trade (2 legs x 2 sides), tick_value = 12.50
        tick_value = 12.50
        fills_per_trade = 4

        # Current cost per trade: slippage + commission
        current_slippage_cost = base_slippage * tick_value * fills_per_trade
        current_commission_cost = base_commission * fills_per_trade
        current_cost_per_trade = current_slippage_cost + current_commission_cost
        total_current_cost = current_cost_per_trade * n_trades

        # Estimate gross P&L
        gross_pnl = total_pnl + total_current_cost

        # 1. Slippage sensitivity (holding commission constant)
        slippage_values = np.arange(0, 5.1, 0.5)
        net_pnl_slippage = []
        for slip in slippage_values:
            slip_cost = slip * tick_value * fills_per_trade * n_trades
            comm_cost = base_commission * fills_per_trade * n_trades
            net = gross_pnl - slip_cost - comm_cost
            net_pnl_slippage.append(net)

        colors = [ACADEMIC_PALETTE["positive"] if p >= 0 else ACADEMIC_PALETTE["negative"]
                  for p in net_pnl_slippage]
        axes[0].bar(slippage_values, net_pnl_slippage, width=0.4, color=colors,
                   edgecolor=ACADEMIC_PALETTE["primary"],
                   linewidth=LAYOUT_SETTINGS["bar_edge_width"], alpha=0.8)
        axes[0].axhline(y=0, color=ACADEMIC_PALETTE["neutral"], linestyle="-",
                       linewidth=1.5)
        axes[0].axvline(x=base_slippage, color=ACADEMIC_PALETTE["accent1"],
                       linestyle="--", linewidth=2, label=f"Current ({base_slippage} ticks)")
        axes[0].set_xlabel("Slippage (ticks)", fontsize=FONT_SETTINGS["label_size"])
        axes[0].set_ylabel("Net P&L ($)", fontsize=FONT_SETTINGS["label_size"])
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        axes[0].legend(fontsize=FONT_SETTINGS["legend_size"], framealpha=0.9)
        apply_style(axes[0], title="Slippage Sensitivity")

        # 2. Commission sensitivity (holding slippage constant)
        commission_values = np.arange(0, 10.1, 1.0)
        net_pnl_commission = []
        for comm in commission_values:
            slip_cost = base_slippage * tick_value * fills_per_trade * n_trades
            comm_cost = comm * fills_per_trade * n_trades
            net = gross_pnl - slip_cost - comm_cost
            net_pnl_commission.append(net)

        colors = [ACADEMIC_PALETTE["positive"] if p >= 0 else ACADEMIC_PALETTE["negative"]
                  for p in net_pnl_commission]
        axes[1].bar(commission_values, net_pnl_commission, width=0.8, color=colors,
                   edgecolor=ACADEMIC_PALETTE["primary"],
                   linewidth=LAYOUT_SETTINGS["bar_edge_width"], alpha=0.8)
        axes[1].axhline(y=0, color=ACADEMIC_PALETTE["neutral"], linestyle="-",
                       linewidth=1.5)
        axes[1].axvline(x=base_commission, color=ACADEMIC_PALETTE["accent1"],
                       linestyle="--", linewidth=2, label=f"Current (${base_commission})")
        axes[1].set_xlabel("Commission ($/contract)", fontsize=FONT_SETTINGS["label_size"])
        axes[1].set_ylabel("Net P&L ($)", fontsize=FONT_SETTINGS["label_size"])
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        axes[1].legend(fontsize=FONT_SETTINGS["legend_size"], framealpha=0.9)
        apply_style(axes[1], title="Commission Sensitivity")

        fig.suptitle(
            f"{symbol} {strategy.upper()} Cost Sensitivity Analysis",
            fontsize=FONT_SETTINGS["title_size"],
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_{strategy}_cost_sensitivity.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def plot_rolling_performance(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy: str = "eom",
        window: int = 20,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create chart showing rolling Sharpe ratio and win rate."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Clean data
        df_clean = df.dropna(subset=["pnl", "exit_date"]).copy()
        if df_clean.empty or len(df_clean) < window:
            plt.close(fig)
            raise ValueError(f"Not enough data for {window}-trade rolling window")

        df_clean = df_clean.sort_values("exit_date")
        df_clean["exit_date"] = pd.to_datetime(df_clean["exit_date"])
        dates = df_clean["exit_date"]
        pnl = df_clean["pnl"]

        # Calculate rolling Sharpe (annualized)
        rolling_mean = pnl.rolling(window=window, min_periods=window).mean()
        rolling_std = pnl.rolling(window=window, min_periods=window).std()
        # Annualize assuming ~252 trading days and estimate trades per day
        avg_trades_per_year = len(df_clean) / ((dates.max() - dates.min()).days / 365.25)
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(avg_trades_per_year)

        # Calculate rolling win rate
        rolling_win_rate = pnl.rolling(window=window, min_periods=window).apply(
            lambda x: (x >= 0).sum() / len(x) * 100, raw=False
        )

        # Plot rolling Sharpe
        axes[0].plot(dates, rolling_sharpe, color=ACADEMIC_PALETTE["primary"],
                    linewidth=LAYOUT_SETTINGS["line_width"], label=f"{window}-Trade Rolling Sharpe")
        axes[0].fill_between(dates, rolling_sharpe, 0, where=(rolling_sharpe >= 0),
                            color=ACADEMIC_PALETTE["positive"], alpha=0.3)
        axes[0].fill_between(dates, rolling_sharpe, 0, where=(rolling_sharpe < 0),
                            color=ACADEMIC_PALETTE["negative"], alpha=0.3)
        axes[0].axhline(y=0, color=ACADEMIC_PALETTE["neutral"], linestyle="-",
                       linewidth=1.5)
        axes[0].axhline(y=1, color=ACADEMIC_PALETTE["positive"], linestyle="--",
                       linewidth=1.5, alpha=0.6, label="Sharpe=1")
        axes[0].axhline(y=-1, color=ACADEMIC_PALETTE["negative"], linestyle="--",
                       linewidth=1.5, alpha=0.6, label="Sharpe=-1")
        apply_style(axes[0], ylabel="Rolling Sharpe Ratio")
        axes[0].legend(loc="upper left", fontsize=FONT_SETTINGS["legend_size"],
                      framealpha=0.9)

        # Plot rolling win rate
        axes[1].plot(dates, rolling_win_rate, color=ACADEMIC_PALETTE["accent2"],
                    linewidth=LAYOUT_SETTINGS["line_width"], label=f"{window}-Trade Rolling Win Rate")
        axes[1].fill_between(dates, rolling_win_rate, 50, where=(rolling_win_rate >= 50),
                            color=ACADEMIC_PALETTE["positive"], alpha=0.3)
        axes[1].fill_between(dates, rolling_win_rate, 50, where=(rolling_win_rate < 50),
                            color=ACADEMIC_PALETTE["negative"], alpha=0.3)
        axes[1].axhline(y=50, color=ACADEMIC_PALETTE["neutral"], linestyle="-",
                       linewidth=1.5, label="50% Win Rate")
        axes[1].set_ylim(0, 100)
        apply_style(axes[1], xlabel="Date", ylabel="Rolling Win Rate (%)")
        axes[1].legend(loc="upper left", fontsize=FONT_SETTINGS["legend_size"],
                      framealpha=0.9)

        fig.suptitle(
            f"{symbol} {strategy.upper()} Rolling Performance ({window}-Trade Window)",
            fontsize=FONT_SETTINGS["title_size"],
            fontweight="bold",
            y=1.02,
        )

        fig.autofmt_xdate()
        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / f"{symbol}_{strategy}_rolling_performance.pdf"

        fig.savefig(output_path, **FIGURE_DEFAULTS)
        plt.close(fig)
        return output_path

    def generate_all(self, symbol: str, trades_dir: Path | str) -> dict[str, Path]:
        """Generate all Stage 4 backtest plots from saved parquet data."""
        trades_dir = Path(trades_dir)
        outputs = {}

        # EOM equity curve
        eom_equity_path = trades_dir / "eom_equity.parquet"
        if eom_equity_path.exists():
            df = pd.read_parquet(eom_equity_path)
            try:
                outputs["eom_equity"] = self.plot_equity_curve(df, symbol, "eom")
            except ValueError:
                pass

        # DTE equity curve
        dte_equity_path = trades_dir / "dte_equity.parquet"
        if dte_equity_path.exists():
            df = pd.read_parquet(dte_equity_path)
            try:
                outputs["dte_equity"] = self.plot_equity_curve(df, symbol, "dte")
            except ValueError:
                pass

        # Strategy comparison
        summary_path = trades_dir / "summary.parquet"
        if summary_path.exists():
            df = pd.read_parquet(summary_path)
            outputs["strategy_comparison"] = self.plot_strategy_comparison(df, symbol)

        # EOM trade P&L distribution and heatmap
        eom_trades_path = trades_dir / "eom_trades.parquet"
        if eom_trades_path.exists():
            df = pd.read_parquet(eom_trades_path)
            try:
                outputs["eom_pnl_dist"] = self.plot_trade_pnl_distribution(df, symbol, "eom")
                outputs["eom_monthly_heatmap"] = self.plot_monthly_returns_heatmap(
                    df, symbol, "eom"
                )
                # New plots
                outputs["eom_cumulative_monthly"] = self.plot_cumulative_pnl_by_month(
                    df, symbol, "eom"
                )
                outputs["eom_cost_sensitivity"] = self.plot_cost_sensitivity(
                    df, symbol, "eom"
                )
                outputs["eom_rolling_performance"] = self.plot_rolling_performance(
                    df, symbol, "eom"
                )
            except ValueError:
                pass

        # DTE trade P&L distribution
        dte_trades_path = trades_dir / "dte_trades.parquet"
        if dte_trades_path.exists():
            df = pd.read_parquet(dte_trades_path)
            try:
                outputs["dte_pnl_dist"] = self.plot_trade_pnl_distribution(df, symbol, "dte")
                outputs["dte_monthly_heatmap"] = self.plot_monthly_returns_heatmap(
                    df, symbol, "dte"
                )
            except ValueError:
                pass

        return outputs

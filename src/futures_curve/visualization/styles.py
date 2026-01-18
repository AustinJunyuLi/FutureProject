"""Visualization styles and color schemes for academic publication-quality figures."""

from typing import Dict, Any, Tuple
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def _configure_fonts() -> None:
    """Select an available font to avoid hard failures in minimal environments."""
    preferred = [
        "DejaVu Sans",
        "Liberation Sans",
        "Arial",
        "Nimbus Sans L",
        "Helvetica",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    if not available:
        return

    chosen = None
    for name in preferred:
        if name in available:
            chosen = name
            break
    if chosen is None:
        chosen = sorted(available)[0]

    mpl.rcParams["font.family"] = chosen
    mpl.rcParams["font.sans-serif"] = [chosen]


_configure_fonts()

# Academic color palette - muted, professional colors
ACADEMIC_PALETTE: Dict[str, str] = {
    "primary": "#2C3E50",       # Dark blue-gray (main lines)
    "secondary": "#7F8C8D",     # Muted gray
    "positive": "#27AE60",      # Muted green (profit/contango)
    "negative": "#C0392B",      # Muted red (loss/backwardation)
    "neutral": "#95A5A6",       # Light gray (reference lines)
    "ci_band": "#4A90A4",       # Steel blue (confidence intervals)
    "accent1": "#8E44AD",       # Purple (accent)
    "accent2": "#E67E22",       # Orange (accent)
    "contango": "#27AE60",      # Green - contango state
    "backwardation": "#C0392B", # Red - backwardation state
}

# Legacy COLORS for backward compatibility
COLORS: Dict[str, str] = {
    "primary": ACADEMIC_PALETTE["primary"],
    "secondary": ACADEMIC_PALETTE["secondary"],
    "positive": ACADEMIC_PALETTE["positive"],
    "negative": ACADEMIC_PALETTE["negative"],
    "neutral": ACADEMIC_PALETTE["neutral"],
    "contango": ACADEMIC_PALETTE["contango"],
    "backwardation": ACADEMIC_PALETTE["backwardation"],
    "ci_band": ACADEMIC_PALETTE["ci_band"],
    "heatmap_cmap": "RdYlGn",
    "diverging_cmap": "RdBu_r",
}

# Default savefig settings (applies to fig.savefig())
FIGURE_DEFAULTS: Dict[str, Any] = {
    "dpi": 300,
    "bbox_inches": "tight",
    "facecolor": "white",
    "edgecolor": "none",
    "pad_inches": 0.1,
}

# Font settings - larger for publication readability
FONT_SETTINGS: Dict[str, Any] = {
    "title_size": 16,
    "subtitle_size": 14,
    "label_size": 12,
    "tick_size": 10,
    "legend_size": 10,
    "annotation_size": 9,
    "heatmap_annot_size": 8,
}

# Layout settings
LAYOUT_SETTINGS: Dict[str, Any] = {
    "spine_linewidth": 1.0,
    "grid_alpha": 0.3,
    "grid_linestyle": "--",
    "line_width": 2.0,
    "marker_size": 6,
    "bar_edge_width": 0.5,
}

# Plot-specific settings
PLOT_SETTINGS: Dict[str, Dict[str, Any]] = {
    "equity_curve": {
        "figsize": (12, 6),
        "linewidth": 2.0,
    },
    "heatmap": {
        "figsize": (14, 6),
        "annot": True,
        "fmt": ".2f",
    },
    "histogram": {
        "figsize": (10, 6),
        "bins": 30,
        "alpha": 0.75,
        "edgecolor": "white",
        "linewidth": 0.5,
    },
    "bar": {
        "figsize": (10, 6),
        "width": 0.35,
    },
    "line": {
        "figsize": (12, 6),
        "linewidth": 2.0,
        "marker": "o",
        "markersize": 6,
    },
    "grid": {
        "figsize": (14, 10),
    },
}


def get_heatmap_text_color(value: float, vmin: float, vmax: float, cmap_name: str = "RdYlGn") -> str:
    """
    Determine text color (black or white) for heatmap annotations based on background.

    Args:
        value: The cell value
        vmin: Minimum value of colormap range
        vmax: Maximum value of colormap range
        cmap_name: Name of the colormap

    Returns:
        Color string: 'white' for dark backgrounds, 'black' for light backgrounds
    """
    if np.isnan(value):
        return "black"

    # Normalize value to [0, 1]
    if vmax == vmin:
        norm_val = 0.5
    else:
        norm_val = (value - vmin) / (vmax - vmin)
    norm_val = np.clip(norm_val, 0, 1)

    # Get colormap and RGBA value
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm_val)

    # Calculate luminance using standard coefficients
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]

    # Return white text for dark backgrounds (luminance < 0.5)
    return "white" if luminance < 0.5 else "black"


def apply_style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent styling to an axis."""
    if title:
        ax.set_title(title, fontsize=FONT_SETTINGS["title_size"], fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SETTINGS["label_size"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SETTINGS["label_size"])
    ax.tick_params(labelsize=FONT_SETTINGS["tick_size"])
    ax.grid(True, alpha=LAYOUT_SETTINGS["grid_alpha"], linestyle=LAYOUT_SETTINGS["grid_linestyle"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(LAYOUT_SETTINGS["spine_linewidth"])
    ax.spines["bottom"].set_linewidth(LAYOUT_SETTINGS["spine_linewidth"])


def apply_academic_style(fig, ax_or_axes) -> None:
    """
    Apply comprehensive academic styling to a figure.

    Args:
        fig: matplotlib Figure object
        ax_or_axes: Single axis or array of axes
    """
    # Handle both single axis and array of axes
    if hasattr(ax_or_axes, '__iter__'):
        axes = ax_or_axes.flatten() if hasattr(ax_or_axes, 'flatten') else list(ax_or_axes)
    else:
        axes = [ax_or_axes]

    for ax in axes:
        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(LAYOUT_SETTINGS["spine_linewidth"])
        ax.spines["bottom"].set_linewidth(LAYOUT_SETTINGS["spine_linewidth"])

        # Grid styling
        ax.grid(True, alpha=LAYOUT_SETTINGS["grid_alpha"],
                linestyle=LAYOUT_SETTINGS["grid_linestyle"],
                color=ACADEMIC_PALETTE["neutral"])

        # Tick styling
        ax.tick_params(labelsize=FONT_SETTINGS["tick_size"], width=LAYOUT_SETTINGS["spine_linewidth"])

    # Tight layout
    fig.tight_layout()


def create_figure(nrows: int = 1, ncols: int = 1,
                  figsize: Tuple[float, float] = None,
                  plot_type: str = "line") -> Tuple[plt.Figure, Any]:
    """
    Create a figure with academic styling defaults.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size tuple (width, height)
        plot_type: Type of plot for default figsize

    Returns:
        Tuple of (figure, axes)
    """
    if figsize is None:
        figsize = PLOT_SETTINGS.get(plot_type, PLOT_SETTINGS["line"])["figsize"]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def format_currency(value: float, decimals: int = 0) -> str:
    """Format a number as currency."""
    if abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a number as percentage."""
    return f"{value:.{decimals}f}%"

#!/usr/bin/env python
"""Generate all figures for Stage 3 and Stage 4 reports."""

from pathlib import Path
from futures_curve.visualization.analysis_plots import AnalysisVisualizer
from futures_curve.visualization.backtest_plots import BacktestVisualizer

def main():
    figures_dir = Path("research_outputs/figures")
    tables_dir = Path("research_outputs/tables")
    trades_dir = Path("data_parquet/trades/HG")

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Stage 3 figures
    print("Generating Stage 3 analysis figures...")
    av = AnalysisVisualizer(figures_dir)
    stage3_outputs = av.generate_all("HG", tables_dir, data_dir=Path("data_parquet"))
    for name, path in stage3_outputs.items():
        print(f"  Created: {path}")

    # Stage 4 figures
    print("\nGenerating Stage 4 backtest figures...")
    bv = BacktestVisualizer(figures_dir)
    stage4_outputs = bv.generate_all("HG", trades_dir)
    for name, path in stage4_outputs.items():
        print(f"  Created: {path}")

    print(f"\nTotal figures generated: {len(stage3_outputs) + len(stage4_outputs)}")

if __name__ == "__main__":
    main()

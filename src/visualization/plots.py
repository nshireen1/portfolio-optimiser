"""
src/visualization/plots.py
---------------------------
Publication-quality visualisations for the portfolio optimisation experiment.

Panels:
  1. Efficient frontier with strategy points
  2. Rolling out-of-sample cumulative wealth
  3. In-sample vs out-of-sample comparison (6-point mean-std diagram)
  4. Rolling Sharpe ratio over time
  5. Weight heatmap (how allocations evolve)
  6. Master figure (all panels combined)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
})

COLORS = {
    "MV":   "#2563EB",   # Blue
    "NSMV": "#059669",   # Green
    "MIMV": "#DC2626",   # Red
}
MARKERS = {"In-Sample": "o", "Out-of-Sample": "s"}
LABELS  = {
    "MV":   "Unconstrained MV",
    "NSMV": "No-Shorting MV",
    "MIMV": "Min-Investment MV",
}


def plot_efficient_frontier(
    frontier_std:  np.ndarray,
    frontier_mean: np.ndarray,
    strategy_points: Dict,   # {strategy: (std, mean)}
    monthly_target: float,
    save: bool = True,
) -> plt.Figure:
    """Efficient frontier curve with strategy operating points."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Frontier curve
    ax.plot(
        frontier_std * 100, frontier_mean * 100,
        color="#6B7280", linewidth=2.5, label="Efficient frontier (MV)",
        zorder=2
    )
    ax.fill_between(
        frontier_std * 100, frontier_mean * 100,
        alpha=0.06, color="#2563EB"
    )

    # Strategy points
    for strategy, (std, mean) in strategy_points.items():
        ax.scatter(
            std * 100, mean * 100,
            color=COLORS[strategy], s=160, zorder=5,
            edgecolors="white", linewidths=2,
            label=LABELS[strategy]
        )
        ax.annotate(
            LABELS[strategy],
            (std * 100, mean * 100),
            xytext=(8, 4), textcoords="offset points",
            fontsize=8, color=COLORS[strategy]
        )

    # Target return line
    ax.axhline(
        monthly_target * 100, linestyle="--",
        color="#D97706", linewidth=1.2, alpha=0.7,
        label=f"Target return ({monthly_target*100:.2f}%/month)"
    )

    ax.set_xlabel("Monthly standard deviation (%)")
    ax.set_ylabel("Monthly mean return (%)")
    ax.set_title(
        "Mean-Variance Efficient Frontier\n"
        "In-sample optimal portfolios for the three strategies",
        pad=10
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "efficient_frontier.png", bbox_inches="tight")
    return fig


def plot_cumulative_wealth(
    oos_results: Dict[str, pd.Series],
    save: bool = True,
) -> plt.Figure:
    """Rolling out-of-sample cumulative wealth — £1 invested at start."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for strategy, series in oos_results.items():
        s   = series.dropna()
        cum = (1 + s).cumprod()
        ax.plot(
            cum.index, cum.values,
            color=COLORS[strategy], linewidth=2,
            label=f"{LABELS[strategy]}  (final: £{cum.iloc[-1]:.2f})"
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth (£1 invested)")
    ax.set_title(
        "Out-of-sample cumulative wealth  ·  Rolling 60-month window\n"
        "Constrained strategies (NSMV, MIMV) outperform unconstrained MV",
        pad=10
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('£%.1f'))

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "cumulative_wealth.png", bbox_inches="tight")
    return fig


def plot_mean_std_diagram(
    is_results:  Dict[str, pd.Series],
    oos_results: Dict[str, pd.Series],
    save: bool = True,
) -> plt.Figure:
    """The classic 6-point mean-std diagram from the report."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for strategy in ["MV", "NSMV", "MIMV"]:
        for eval_type, results in [("In-Sample", is_results), ("Out-of-Sample", oos_results)]:
            s    = results[strategy].dropna()
            mean = s.mean() * 100
            std  = s.std()  * 100
            ax.scatter(
                std, mean,
                color=COLORS[strategy],
                marker=MARKERS[eval_type],
                s=160, zorder=4,
                edgecolors="white", linewidths=1.5,
                alpha=0.9
            )
            offset = (3, 4) if eval_type == "Out-of-Sample" else (-3, -12)
            ax.annotate(
                f"{strategy}\n({eval_type[:2]})",
                (std, mean),
                xytext=offset, textcoords="offset points",
                fontsize=7.5, color=COLORS[strategy], ha="center"
            )

    # Legend
    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#374151", markersize=9, label="In-Sample"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#374151", markersize=9, label="Out-of-Sample"),
    ] + [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=COLORS[s], markersize=9, label=LABELS[s])
        for s in COLORS
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    ax.set_xlabel("Monthly standard deviation (%)")
    ax.set_ylabel("Monthly mean return (%)")
    ax.set_title(
        "Mean-Standard Deviation Diagram  ·  6 Portfolio Strategies\n"
        "Out-of-sample points (■) shift right but constrained strategies earn more",
        pad=10
    )
    ax.grid(alpha=0.25)

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "mean_std_diagram.png", bbox_inches="tight")
    return fig


def plot_rolling_sharpe(
    oos_results: Dict[str, pd.Series],
    window: int = 24,
    save: bool = True,
) -> plt.Figure:
    """Rolling 24-month Sharpe ratio for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for strategy, series in oos_results.items():
        s = series.dropna()
        rolling_sharpe = (
            s.rolling(window).mean() / s.rolling(window).std()
        ) * np.sqrt(12)   # annualised

        ax.plot(
            rolling_sharpe.index, rolling_sharpe.values,
            color=COLORS[strategy], linewidth=1.8,
            label=LABELS[strategy]
        )

    ax.axhline(0, color="#9CA3AF", linewidth=1, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised Sharpe ratio (rolling 24m)")
    ax.set_title(
        f"Rolling {window}-month Sharpe ratio  ·  Out-of-Sample\n"
        "Constrained strategies maintain higher risk-adjusted returns",
        pad=10
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "rolling_sharpe.png", bbox_inches="tight")
    return fig


def plot_performance_table(
    summary_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Visual performance table with colour coding."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    cols = ["Strategy", "Evaluation", "Mean Return", "Std Dev", "Sharpe", "Annual Return"]
    disp = summary_df[["strategy","eval","mean","std","sharpe","annual_return"]].copy()
    disp.columns = cols
    disp["Mean Return"]   = disp["Mean Return"].apply(lambda x: f"{x*100:.4f}%")
    disp["Std Dev"]       = disp["Std Dev"].apply(lambda x: f"{x*100:.4f}%")
    disp["Sharpe"]        = disp["Sharpe"].apply(lambda x: f"{x:.4f}")
    disp["Annual Return"] = disp["Annual Return"].apply(lambda x: f"{x*100:.2f}%")

    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Header styling
    for j in range(len(cols)):
        table[0, j].set_facecolor("#2563EB")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row colour coding
    for i in range(1, len(disp) + 1):
        for j in range(len(cols)):
            table[i, j].set_facecolor("#F9FAFB" if i % 2 == 0 else "white")

    ax.set_title("Portfolio Performance Summary", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "performance_table.png", bbox_inches="tight")
    return fig


def plot_master_figure(
    frontier_std:    np.ndarray,
    frontier_mean:   np.ndarray,
    strategy_points: Dict,
    monthly_target:  float,
    is_results:      Dict[str, pd.Series],
    oos_results:     Dict[str, pd.Series],
    summary_df:      pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Single publication-quality master figure — this goes in your README and LinkedIn."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Portfolio Optimisation: Constraint Regularisation in Mean-Variance Optimisation\n"
        "Developed 25 Portfolios (Fama-French)  ·  Sep 1990 – Oct 2023  ·  Rolling 60-month OOS",
        fontsize=12, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])   # Efficient frontier
    ax2 = fig.add_subplot(gs[0, 1])   # Mean-std diagram
    ax3 = fig.add_subplot(gs[0, 2])   # OOS bar chart
    ax4 = fig.add_subplot(gs[1, :2])  # Cumulative wealth (wide)
    ax5 = fig.add_subplot(gs[1, 2])   # Rolling Sharpe

    # ── A: Efficient frontier ─────────────────────────────────────────────
    ax1.plot(frontier_std * 100, frontier_mean * 100,
             color="#6B7280", linewidth=2, label="Frontier")
    ax1.fill_between(frontier_std * 100, frontier_mean * 100, alpha=0.07, color="#2563EB")
    for s, (std, mean) in strategy_points.items():
        ax1.scatter(std*100, mean*100, color=COLORS[s], s=120,
                    edgecolors="white", linewidths=1.5, zorder=4, label=LABELS[s])
    ax1.axhline(monthly_target*100, linestyle="--", color="#D97706", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Std Dev (%/month)")
    ax1.set_ylabel("Mean Return (%/month)")
    ax1.set_title("A. Efficient Frontier", fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.2)

    # ── B: Mean-std 6-point diagram ───────────────────────────────────────
    for strategy in ["MV", "NSMV", "MIMV"]:
        for eval_type, results in [("In-Sample", is_results), ("Out-of-Sample", oos_results)]:
            s    = results[strategy].dropna()
            mean = s.mean() * 100
            std  = s.std()  * 100
            ax2.scatter(std, mean, color=COLORS[strategy],
                        marker=MARKERS[eval_type], s=130, zorder=4,
                        edgecolors="white", linewidths=1.5, alpha=0.9)
            ax2.annotate(f"{strategy}({'IS' if eval_type=='In-Sample' else 'OOS'})",
                         (std, mean), xytext=(3, 3), textcoords="offset points",
                         fontsize=7, color=COLORS[strategy])
    ax2.set_xlabel("Std Dev (%/month)")
    ax2.set_ylabel("Mean Return (%/month)")
    ax2.set_title("B. Mean-Std Diagram (IS vs OOS)", fontweight="bold")
    ax2.grid(alpha=0.2)

    # ── C: OOS performance bar chart ─────────────────────────────────────
    oos_means  = [oos_results[s].dropna().mean()*100 for s in ["MV","NSMV","MIMV"]]
    oos_sharpe = [oos_results[s].dropna().mean()/oos_results[s].dropna().std()*np.sqrt(12)
                  for s in ["MV","NSMV","MIMV"]]
    x = np.arange(3)
    bars = ax3.bar(x, oos_means, color=[COLORS[s] for s in ["MV","NSMV","MIMV"]],
                   alpha=0.9, width=0.5, zorder=2)
    for bar, m in zip(bars, oos_means):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{m:.3f}%", ha="center", fontsize=9, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["MV", "NSMV", "MIMV"])
    ax3.set_ylabel("Mean monthly return (%)")
    ax3.set_title("C. OOS Mean Return\nConstraints beat unconstrained", fontweight="bold")
    ax3.grid(axis="y", alpha=0.2, zorder=1)

    # ── D: Cumulative wealth ──────────────────────────────────────────────
    for strategy, series in oos_results.items():
        s   = series.dropna()
        cum = (1 + s).cumprod()
        ax4.plot(cum.index, cum.values, color=COLORS[strategy], linewidth=2,
                 label=f"{LABELS[strategy]}  (£{cum.iloc[-1]:.2f})")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Cumulative wealth (£1 invested)")
    ax4.set_title("D. Out-of-Sample Cumulative Wealth  ·  Rolling 60-month window", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.2)

    # ── E: Rolling Sharpe ─────────────────────────────────────────────────
    for strategy, series in oos_results.items():
        s = series.dropna()
        rs = (s.rolling(24).mean() / s.rolling(24).std()) * np.sqrt(12)
        ax5.plot(rs.index, rs.values, color=COLORS[strategy], linewidth=1.5,
                 label=LABELS[strategy])
    ax5.axhline(0, color="#9CA3AF", linewidth=1, linestyle="--")
    ax5.set_xlabel("Date")
    ax5.set_ylabel("Ann. Sharpe (rolling 24m)")
    ax5.set_title("E. Rolling Sharpe Ratio", fontweight="bold")
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.2)

    if save:
        fig.savefig(OUTPUT_DIR / "master_figure.png", bbox_inches="tight", dpi=200)
    return fig

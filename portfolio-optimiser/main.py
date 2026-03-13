"""
main.py
-------
Full pipeline for the portfolio optimisation experiment.

Reproduces and extends the IB99L0 Financial Analytics assignment results:
  - Three strategies: MV, NSMV, MIMV
  - In-sample and out-of-sample evaluation
  - Efficient frontier
  - Rolling Sharpe ratio
  - Cumulative wealth curves

Run:
    python main.py
"""

import logging
import sys
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import (
    OUTPUT_DIR, DATA_DIR, MONTHLY_TARGET, M_WINDOW,
    STRATEGIES, STRATEGY_NAMES, ANNUAL_TARGET
)
from src.data.loader import load_returns, make_synthetic_returns
from src.models.optimiser import (
    run_in_sample, run_out_of_sample,
    performance_metrics, build_efficient_frontier, optimise, estimate_params
)
from src.visualization.plots import (
    plot_efficient_frontier,
    plot_cumulative_wealth,
    plot_mean_std_diagram,
    plot_rolling_sharpe,
    plot_performance_table,
    plot_master_figure,
)
import pandas as pd


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # ── 1. Data ───────────────────────────────────────────────────────────
    logger.info("="*55)
    logger.info("  STEP 1: Data")
    logger.info("="*55)

    data_path = DATA_DIR / "Developed_25_Portfolios_ME_BE-ME.csv"
    if data_path.exists():
        logger.info("Loading real Fama-French data...")
        df = load_returns(data_path)
    else:
        logger.info("Real data not found — using synthetic demo data.")
        logger.info("Download from: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html")
        df = make_synthetic_returns()

    # ── 2. In-sample ──────────────────────────────────────────────────────
    logger.info("="*55)
    logger.info("  STEP 2: In-sample evaluation")
    logger.info("="*55)
    is_results = run_in_sample(df, MONTHLY_TARGET)

    # ── 3. Out-of-sample ──────────────────────────────────────────────────
    logger.info("="*55)
    logger.info("  STEP 3: Out-of-sample evaluation (rolling %d months)", M_WINDOW)
    logger.info("="*55)
    oos_results = run_out_of_sample(df, M_WINDOW, MONTHLY_TARGET)

    # ── 4. Performance summary ────────────────────────────────────────────
    logger.info("="*55)
    logger.info("  STEP 4: Performance metrics")
    logger.info("="*55)

    rows = []
    for eval_type, results in [("In-Sample", is_results), ("Out-of-Sample", oos_results)]:
        for strategy in STRATEGIES:
            m = performance_metrics(results[strategy])
            rows.append({"strategy": STRATEGY_NAMES[strategy], "eval": eval_type, **m})
    summary_df = pd.DataFrame(rows)

    # ── 5. Efficient frontier ─────────────────────────────────────────────
    logger.info("="*55)
    logger.info("  STEP 5: Efficient frontier")
    logger.info("="*55)
    frontier_std, frontier_mean = build_efficient_frontier(df)

    # In-sample strategy operating points
    mu, sigma = estimate_params(df.values)
    strategy_points = {}
    for s in STRATEGIES:
        w = optimise(mu, sigma, MONTHLY_TARGET, s)
        if not np.isnan(w).any():
            strategy_points[s] = (
                np.sqrt(w @ sigma @ w),
                w @ mu,
            )

    # ── 6. Visualisations ─────────────────────────────────────────────────
    logger.info("="*55)
    logger.info("  STEP 6: Generating visualisations")
    logger.info("="*55)

    plot_efficient_frontier(frontier_std, frontier_mean, strategy_points, MONTHLY_TARGET)
    logger.info("  ✓ Efficient frontier")

    plot_cumulative_wealth(oos_results)
    logger.info("  ✓ Cumulative wealth")

    plot_mean_std_diagram(is_results, oos_results)
    logger.info("  ✓ Mean-std diagram")

    plot_rolling_sharpe(oos_results)
    logger.info("  ✓ Rolling Sharpe")

    plot_performance_table(summary_df)
    logger.info("  ✓ Performance table")

    plot_master_figure(
        frontier_std, frontier_mean, strategy_points, MONTHLY_TARGET,
        is_results, oos_results, summary_df
    )
    logger.info("  ✓ Master figure")

    # ── 7. Results summary ────────────────────────────────────────────────
    logger.info("="*55)
    logger.info("  RESULTS SUMMARY")
    logger.info("="*55)

    print(f"\n{'═'*70}")
    print(f"  {'Strategy':<22} {'Eval':<14} {'Mean/mo':>9} {'Std/mo':>9} {'Sharpe':>8} {'Annual':>9}")
    print(f"{'─'*70}")
    for _, row in summary_df.iterrows():
        print(
            f"  {row['strategy']:<22} {row['eval']:<14} "
            f"{row['mean']*100:>8.4f}% {row['std']*100:>8.4f}% "
            f"{row['sharpe']:>8.4f} {row['annual_return']*100:>8.2f}%"
        )
    print(f"{'═'*70}")

    oos_mv   = performance_metrics(oos_results["MV"])
    oos_mimv = performance_metrics(oos_results["MIMV"])
    improvement = (oos_mimv["mean"] - oos_mv["mean"]) / abs(oos_mv["mean"]) * 100
    print(f"\n  MIMV vs MV out-of-sample mean return improvement: +{improvement:.1f}%")
    print(f"  All plots saved to outputs/")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()

"""
src/models/optimiser.py
------------------------
Mean-variance portfolio optimisation with three constraint regimes.

Three strategies
-----------------
MV   — Unconstrained mean-variance. Short selling permitted.
       Minimises w'Σw subject to: Σw=1, w'μ=μ_target
       Classic Markowitz (1952) formulation.

NSMV — No-shorting constraint. wi ≥ 0 for all i.
       Prevents extreme short positions driven by estimation error.
       Shown by Jagannathan & Ma (2003) to reduce out-of-sample risk.

MIMV — Minimum investment constraint. wi ≥ 1/(2N) for all i.
       Forces diversification: each asset gets at least 2% allocation.
       Strongest regularisation — best out-of-sample return in our results.

Key insight
-----------
In-sample, MV wins (by construction — it's the unconstrained optimum).
Out-of-sample, NSMV and MIMV win — constraints act as regularisation,
preventing the optimizer from over-fitting to estimation noise.

This is "error maximization" (Michaud, 1989) in practice.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import (
    MONTHLY_TARGET, M_WINDOW, STRATEGIES, N_ASSETS
)

logger = logging.getLogger(__name__)


def estimate_params(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unbiased sample mean and covariance."""
    mu    = np.mean(returns, axis=0)
    sigma = np.cov(returns, rowvar=False, ddof=1)
    return mu, sigma


def optimise(
    mu:       np.ndarray,
    sigma:    np.ndarray,
    r_target: float,
    strategy: str,
) -> np.ndarray:
    """
    Solve the mean-variance optimisation problem.

    min  w'Σw
    s.t. Σwi = 1  (fully invested)
         w'μ = r_target  (target return)
         [strategy-specific bounds]

    Returns optimal weights, or NaN array if optimisation fails.
    """
    n  = len(mu)
    w0 = np.ones(n) / n

    # Regularise covariance if not positive definite
    eigvals = np.linalg.eigvals(sigma)
    if eigvals.min() < -1e-8:
        sigma = sigma + np.eye(n) * (-eigvals.min() + 1e-7)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ mu - r_target},
    ]

    if strategy == "MV":
        bounds = [(None, None)] * n
    elif strategy == "NSMV":
        bounds = [(0, None)] * n
    elif strategy == "MIMV":
        min_w  = 1 / (2 * n)
        bounds = [(min_w, None)] * n
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    result = minimize(
        lambda w: w.T @ sigma @ w,
        w0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 2000},
    )

    if result.success:
        return result.x
    else:
        logger.debug("Optimisation failed for %s: %s", strategy, result.message)
        return np.full(n, np.nan)


def run_in_sample(
    df: pd.DataFrame,
    r_target: float = MONTHLY_TARGET,
) -> Dict[str, pd.Series]:
    """
    In-sample evaluation: estimate parameters from FULL dataset,
    apply fixed weights across entire return history.

    Returns dict of {strategy: return_series}
    """
    mu, sigma = estimate_params(df.values)
    results   = {}

    for strategy in STRATEGIES:
        weights = optimise(mu, sigma, r_target, strategy)
        if np.isnan(weights).any():
            results[strategy] = pd.Series(dtype=float)
            continue

        realized = []
        for t in range(len(df) - 1):
            realized.append(weights @ df.iloc[t + 1].values)
        results[strategy] = pd.Series(realized, index=df.index[1:])

    logger.info("In-sample evaluation complete.")
    return results


def run_out_of_sample(
    df: pd.DataFrame,
    m: int    = M_WINDOW,
    r_target: float = MONTHLY_TARGET,
) -> Dict[str, pd.Series]:
    """
    Out-of-sample rolling window evaluation.

    At each time t (starting from t=M):
      1. Estimate μ, Σ from the previous M months
      2. Solve optimisation for each strategy
      3. Record the realised return at t+1

    This is the HONEST performance evaluation — the model never
    sees future data. This is what would happen in live trading.
    """
    results = {s: [] for s in STRATEGIES}
    dates   = []

    for t in range(m, len(df)):
        window = df.iloc[t - m : t]
        mu, sigma = estimate_params(window.values)

        next_returns = df.iloc[t].values
        dates.append(df.index[t])

        for strategy in STRATEGIES:
            weights = optimise(mu, sigma, r_target, strategy)
            if np.isnan(weights).any():
                results[strategy].append(np.nan)
            else:
                results[strategy].append(weights @ next_returns)

    logger.info("Out-of-sample evaluation complete (%d periods).", len(dates))
    return {s: pd.Series(results[s], index=dates) for s in STRATEGIES}


def performance_metrics(series: pd.Series) -> Dict[str, float]:
    """Compute mean, std, Sharpe (rf=0), max drawdown, annualised return."""
    s = series.dropna()
    if len(s) < 2:
        return {k: np.nan for k in ["mean", "std", "sharpe", "annual_return", "max_drawdown"]}

    mean    = s.mean()
    std     = s.std()
    sharpe  = mean / std if std > 0 else np.nan
    annual  = (1 + mean) ** 12 - 1

    # Max drawdown on cumulative returns
    cum     = (1 + s).cumprod()
    rolling_max = cum.cummax()
    drawdown    = (cum - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    return {
        "mean":          round(mean,   6),
        "std":           round(std,    6),
        "sharpe":        round(sharpe, 4),
        "annual_return": round(annual, 4),
        "max_drawdown":  round(max_dd, 4),
    }


def build_efficient_frontier(
    df: pd.DataFrame,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the efficient frontier by solving MV optimisation across
    a range of target returns. Returns (std_devs, mean_returns).
    """
    mu, sigma = estimate_params(df.values)
    min_ret   = mu.min() * 1.05
    max_ret   = mu.max() * 0.95

    targets = np.linspace(min_ret, max_ret, n_points)
    frontier_std  = []
    frontier_mean = []

    for r_t in targets:
        w = optimise(mu, sigma, r_t, "MV")
        if not np.isnan(w).any():
            port_var  = w @ sigma @ w
            port_mean = w @ mu
            frontier_std.append(np.sqrt(port_var))
            frontier_mean.append(port_mean)

    return np.array(frontier_std), np.array(frontier_mean)

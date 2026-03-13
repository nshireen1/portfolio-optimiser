"""
src/data/loader.py
------------------
Loads and cleans the Fama-French Developed 25 Portfolios dataset.

The raw CSV from Kenneth R. French's data library has a specific format:
  - Multiple sections separated by blank lines
  - Missing values coded as -99.99
  - Returns expressed as percentages

This module handles all that complexity and returns a clean DataFrame.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SKIPROWS, NROWS, N_ASSETS

logger = logging.getLogger(__name__)


def load_returns(filepath: Path | str) -> pd.DataFrame:
    """
    Load and clean the Fama-French Developed 25 Portfolios CSV.

    Parameters
    ----------
    filepath : path to the raw CSV file

    Returns
    -------
    DataFrame with DatetimeIndex and 25 asset return columns (decimal, not %)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df_chunk = pd.read_csv(
        filepath, skiprows=SKIPROWS, nrows=NROWS + 1,
        header=None, skipinitialspace=True
    )

    # First row of chunk is the header
    header    = df_chunk.iloc[0].astype(str).str.strip()
    df        = df_chunk[1:].copy()
    df.columns = header

    date_col = df.columns[0]
    df[date_col] = df[date_col].astype(str).str.strip()
    df = df[df[date_col].str.match(r'^\d{6}$')]
    df["Date"] = pd.to_datetime(df[date_col], format="%Y%m")
    df = df.set_index("Date")
    if date_col in df.columns:
        df = df.drop(columns=[date_col])

    # Convert to numeric, replace missing values, convert % → decimal
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-99.99, np.nan)
        df[col] = df[col] / 100.0

    df = df.dropna()

    logger.info(
        "Loaded %d months × %d assets  (%s to %s)",
        df.shape[0], df.shape[1],
        df.index.min().strftime("%Y-%m"),
        df.index.max().strftime("%Y-%m"),
    )

    if df.shape[1] != N_ASSETS:
        logger.warning("Expected %d assets but got %d", N_ASSETS, df.shape[1])

    return df


def make_synthetic_returns(n_months: int = 398, n_assets: int = 25) -> pd.DataFrame:
    """
    Generate synthetic monthly returns that mirror Fama-French portfolio
    characteristics — used for demo mode when real data is unavailable.

    Returns are drawn from a multivariate normal with realistic
    mean/covariance structure inspired by actual Fama-French statistics.
    """
    np.random.seed(42)

    # Realistic monthly return parameters (calibrated to 1990-2023 data)
    # Small-value portfolios have higher mean returns but higher volatility
    means = np.linspace(0.004, 0.012, n_assets)   # 0.4% – 1.2% monthly
    vols  = np.linspace(0.030, 0.065, n_assets)   # 3% – 6.5% monthly std

    # Build realistic correlation matrix with size/value factor structure
    corr = np.full((n_assets, n_assets), 0.65)    # Base correlation
    np.fill_diagonal(corr, 1.0)
    # Adjacent portfolios more correlated
    for i in range(n_assets):
        for j in range(n_assets):
            corr[i, j] = 0.65 + 0.30 * np.exp(-abs(i - j) / 5)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)

    # Ensure positive definite
    eigvals = np.linalg.eigvals(corr)
    if eigvals.min() < 0:
        corr += np.eye(n_assets) * (-eigvals.min() + 1e-6)

    cov = np.diag(vols) @ corr @ np.diag(vols)

    returns = np.random.multivariate_normal(means, cov, n_months)

    dates = pd.date_range("1990-09-01", periods=n_months, freq="MS")
    cols  = [f"P{i+1:02d}" for i in range(n_assets)]
    df    = pd.DataFrame(returns, index=dates, columns=cols)

    logger.info("Generated synthetic data: %d months × %d assets", n_months, n_assets)
    return df

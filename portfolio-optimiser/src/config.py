"""
config.py — Central configuration for the Portfolio Optimisation experiment.

Dataset: Developed 25 Portfolios Formed on Size and Book-to-Market (ME/BE-ME)
Source:  Kenneth R. French Data Library
Period:  September 1990 – October 2023 (398 monthly observations)
"""

from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Data loading parameters (specific to the French CSV format)
SKIPROWS       = 21
NROWS          = 399
N_ASSETS       = 25

# Optimisation parameters
ANNUAL_TARGET  = 0.08                             # 8% annual target return
MONTHLY_TARGET = (1 + ANNUAL_TARGET) ** (1/12) - 1  # ≈ 0.6434% per month

# Out-of-sample rolling window
M_WINDOW       = 60   # 5-year estimation window (DeMiguel et al., 2009)

STRATEGIES     = ["MV", "NSMV", "MIMV"]
STRATEGY_NAMES = {
    "MV":   "Unconstrained MV",
    "NSMV": "No-Shorting MV",
    "MIMV": "Min-Investment MV",
}

RANDOM_SEED = 42

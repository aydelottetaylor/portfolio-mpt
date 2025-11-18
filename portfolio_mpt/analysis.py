import pandas as pd
import numpy as np


# Get mean and cov of returns
def mean_cov(returns: pd.DataFrame):
    mu = returns.mean()
    sigma = returns.cov()
    return mu, sigma


# Calculate sharpe ratio from returns
def sharpe(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if isinstance(risk_free, (int, float)):
        rf_per_period = risk_free / periods_per_year
        excess = r - rf_per_period
    else:
        # If a Series is provided, treat it as per-period rf aligned to `returns`
        rf_series = pd.Series(risk_free).reindex(r.index).ffill()
        excess = r - rf_series

    vol = excess.std(ddof=1)
    return (np.sqrt(periods_per_year) * excess.mean() / vol) if vol > 0 else float('nan')

# portfolio_mpt/wrangle.py
from __future__ import annotations

import numpy as np
import pandas as pd

def to_returns(prices, method = 'simple', benchmark = None):
    """
    Compute returns from price data
    """
    
    if method == "simple":
        rets = prices.pct_change().dropna()
    elif method == "log":
        rets = np.log(prices / prices.shift(1)).dropna()
    elif method == "excess":
        if benchmark is None:
            raise ValueError("Benchmark required for excess return")
        if isinstance(benchmark, str):
            bench_rets = prices[benchmark].pct_change()
        else:
            bench_rets = benchmark.pct_change()
        rets = prices.pct_change().sub(bench_rets, axis=0).dropna() 
    else:
        raise ValueError("Invalid method")
    return rets


def cumulative_returns(returns):
    return (1 + returns).cumprod() - 1


def rolling_returns(returns, window = 252):
    return (1 + returns).rolling(window).apply(lambda x: x.prod() - 1)


def log_to_simple(log_returns):
    return np.expm1(log_returns)


def simple_to_log(simple_returns):
    if isinstance(simple_returns, pd.DataFrame):
        bad = (1 + simple_returns <= 0).any().any()
    else:  # Series / array-like
        bad = (1 + simple_returns <= 0).any()
    if bad:
        raise ValueError("simple returns ≤ -100% cannot be converted to log returns")
    
    return np.log1p(simple_returns)
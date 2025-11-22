from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List
from scipy.optimize import minimize

@dataclass
class OptResult:
    weights: np.ndarray
    ret: float
    vol: float
    sharpe: float
    success: bool
    message: str
    
    
def _bounds_long_only(n: int) -> Tuple[Tuple[float, float], ...]:
    return tuple((0.0, 1.0) for _ in range(n))


def _sum_to_one_constraint():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _risk(w: np.ndarray, Sigma: pd.DataFrame) -> float:
    return float(np.sqrt(np.dot(w, Sigma.values @ w)))


def _ret(w: np.ndarray, mu:pd.Series) -> float:
    return float(np.dot(w, mu.values))


def _solve(
    x0: np.ndarray,
    objective,
    args=(),
    bounds = None,
    constraints=(),
):
    return minimize(
        objective, x0 = x0, args = args, bounds = bounds, constraints = constraints, method="SLSQP"
    )


def global_min_variance(mu: pd.Series, Sigma: pd.DataFrame) -> OptResult:
    n = len(mu)
    x0 = np.ones(n) / n
    cons = [_sum_to_one_constraint()]
    bounds = _bounds_long_only(n)
    
    def obj(w, S): # minimize volatility
        # direct minimize variance (square of vol) for smoothness
        return float(np.dot(w, S.values @ w))
    
    res = _solve(x0, obj, args=(Sigma,), bounds = bounds, constraints = cons)
    w = res.x
    vol = _risk(w, Sigma)
    r = _ret(w, mu)
    shp = r / vol if vol > 0 else np.nan
    
    return OptResult(weights=w, ret=r, vol=vol, sharpe=shp, success=res.success, message=res.message)


def optimize_for_target_return(mu: pd.Series, Sigma: pd.DataFrame, target: float) -> OptResult:
    n = len(mu)
    x0 = np.ones(n) / n
    cons = [
        _sum_to_one_constraint(),
        {"type": "eq", "fun": lambda w, m, t: np.dot(w, m.values) - t, "args": (mu, target)},
    ]
    bounds = _bounds_long_only(n)
    
    def obj(w, S):
        return float(np.dot(w, S.values @ w)) # variance
    
    low, high = float(mu.min()), float(mu.max())
    if target < low - 1e-10 or target > high + 1e-10:
        raise ValueError(f"Infeasible target {round(target, 6)}; must be between {round(low, 6)} and {round(high, 6)}")
    
    res = _solve(x0, obj, args=(Sigma,), bounds = bounds, constraints = cons)
    w = res.x
    vol = _risk(w, Sigma)
    r = _ret(w, mu)
    sharpe = r / vol if vol > 0 else np.nan
    return OptResult(w, r, vol, sharpe, res.success, res.message)


def max_sharpe(mu: pd.Series, Sigma: pd.DataFrame, rf: float = 0.0) -> OptResult:
    n = len(mu)
    x0 = np.ones(n) / n
    cons = [_sum_to_one_constraint()]
    bounds = _bounds_long_only(n)
    
    def neg_sharpe(w, m, S, rf_):
        r = float(np.dot(w, m.values))
        v = float(np.sqrt(np.dot(w, S.values @ w)))
        if v <= 0:
            return 1e6
        return - (r - rf_) / v

    res = _solve(x0, neg_sharpe, args=(mu, Sigma, rf), bounds=bounds, constraints=cons)
    w = res.x
    vol = _risk(w, Sigma)
    r = _ret(w, mu)
    sharpe = (r - rf) / vol if vol > 0 else np.nan
    return OptResult(w, r, vol, sharpe, res.success, res.message)


def efficient_frontier(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    n_points: int = 30,
    ret_min: float | None = None, 
    ret_max: float | None = None,
) -> List[OptResult]:
    """
    Returns a list of OptResult along targets covering [ret_min, ret_max].
    Ensures monotone volatility as target increases (convex frontier).
    """
    low = float(mu.min()) if ret_min is None else ret_min
    high = float(mu.max()) if ret_max is None else ret_max
    
    if high <= low:
        raise ValueError("ret_max must be greater than ret_min")
    
    targets = np.linspace(low, high, n_points)
    results = []
    last_vol = -np.inf
    
    for t in targets:
        res = optimize_for_target_return(mu, Sigma, t)
        
        if res.vol < last_vol - 1e-8:
            res.vol = last_vol
        last_vol = res.vol
        results.append(res)
        
    return results

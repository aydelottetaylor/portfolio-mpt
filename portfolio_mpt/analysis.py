from __future__ import annotations
import pandas as pd
import numpy as np



def _validate_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate given return data is usable
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("Returns (returns_df) must be of pandas.DataFrame")
    
    if returns_df.isnull().any().any():
        returns_df = returns_df.dropna(how="any")
        
    if returns_df.shape[1] < 1:
        raise ValueError("Returns (returns_df) must have at least one asset column")
    
    return returns_df


def expected_returns(
    returns_df: pd.DataFrame, 
    method: str = "mean", 
    span: int | None = None, 
    annualize: bool = True, 
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate expected returns per asset.
    
    method = "mean": arithmetic mean of period returns
    method = "ema": exponentially-weighted mean (requires span)
    """
    
    rets = _validate_returns(returns_df)
    
    if method == "mean":
        mu = rets.mean()
    elif method == "ema":
        if span is None:
            raise ValueError("EMA method requires 'span'")
        mu = rets.ewm(span=span, adjust = False).mean().iloc[-1]
    else:
        raise ValueError("method must be either mean or ema")
    
    if annualize:
        mu = mu * periods_per_year
        
    return mu


def covariance(
    returns_df: pd.DataFrame,
    method: str = "sample",
    annualize: bool = True,
    periods_per_year = 252
) -> pd.DataFrame:
    """
    Covariance of asset returns. 
    
    method = "sample": sample covariance
    method = "ledoit_wolf": shrinkage (uses scikit-learn if available)
    """
    rets = _validate_returns(returns_df)
    
    if method == "sample":
        Sigma = rets.cov()
    elif method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(rets.values)
            Sigma = pd.DataFrame(lw.covariance_, index=rets.columns, columns=rets.columns)
        except Exception as e:
            if e.__class__.__name__ == "ImportError":
                raise ImportError(
                    "Ledoit-Wolf requires scikit-learn. Install 'scikit-learn' or use method='sample'."
                )
    else:
        raise ValueError("method must be 'sample' or 'ledoit_wolf'")
    
    if annualize:
        Sigma = Sigma * periods_per_year
        
    return Sigma

def portfolio_metrics(
    weights: np.ndarray | pd.Series,
    mu: pd.Series, 
    Sigma: pd.DataFrame, 
    rf: float = 0.0
) -> dict:
    """
    Creates a dict with {'ret', 'vol', 'sharpe'} using annualized inputs.
    """
    
    ws = np.asarray(weights).reshape(-1)
    if abs(ws.sum() - 1) > 1e-6:
        raise ValueError("weights must sum to 1")
    
    mu = mu.loc[Sigma.index]
    ret = float(np.dot(ws, mu.values))
    vol = float(np.sqrt(np.dot(ws, Sigma.values @ ws)))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    
    return {'ret': ret, 'vol': vol, 'sharpe': sharpe}


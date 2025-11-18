import pandas as pd


def to_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    if method == 'log':
        df = (prices.pct_change().add(1))
        return df.where(~df.isna(), None)
    
    # Base arithmetic
    return prices.pct_change()
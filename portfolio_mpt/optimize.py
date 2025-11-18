import numpy as np
import pandas as pd

# Get the global minimum variance weights
def gmv(cov_matrix: pd.DataFrame) -> pd.Series:
    inv_cov_matrix = np.linalg.pinv(cov_matrix.values)
    weights = inv_cov_matrix.sum(axis=1) / inv_cov_matrix.sum()
    return pd.Series(weights, index=cov_matrix.columns)

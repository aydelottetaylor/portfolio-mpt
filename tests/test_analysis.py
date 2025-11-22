import numpy as np
import pandas as pd
import pytest

import portfolio_mpt.analysis as analysis


# ---------- Fixtures ----------
@pytest.fixture
def returns_df():
    # Small, deterministic per-period returns with one NaN row to exercise dropna
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    data = {
        "A": [0.01, 0.02, np.nan, -0.01, 0.00, 0.03],
        "B": [0.00, 0.01, np.nan,  0.02, -0.01, 0.01],
    }
    df = pd.DataFrame(data, index=idx)
    return df


@pytest.fixture
def cleaned_returns(returns_df):
    # analysis._validate_returns drops NaNs via the public functions
    return returns_df.dropna(how="any")


# ---------- expected_returns ----------
def test_expected_returns_mean_annualized(returns_df):
    mu = analysis.expected_returns(returns_df, method="mean", annualize=True, periods_per_year=252)
    # Manual reference equals mean of non-NaN rows times 252
    ref = returns_df.dropna().mean() * 252
    pd.testing.assert_series_equal(mu, ref)


def test_expected_returns_mean_not_annualized(returns_df):
    mu = analysis.expected_returns(returns_df, method="mean", annualize=False)
    ref = returns_df.dropna().mean()
    pd.testing.assert_series_equal(mu, ref)


def test_expected_returns_ema(returns_df):
    span = 3
    mu = analysis.expected_returns(returns_df, method="ema", span=span, annualize=True, periods_per_year=252)
    # Reference: last row of EWM mean on cleaned data, then annualize
    ref = returns_df.dropna().ewm(span=span, adjust=False).mean().iloc[-1] * 252
    pd.testing.assert_series_equal(mu, ref)


def test_expected_returns_invalid_method(returns_df):
    with pytest.raises(ValueError):
        analysis.expected_returns(returns_df, method="median")


def test_expected_returns_ema_requires_span(returns_df):
    with pytest.raises(ValueError):
        analysis.expected_returns(returns_df, method="ema", span=None)


# ---------- covariance ----------
def test_covariance_sample_annualized(returns_df):
    Sigma = analysis.covariance(returns_df, method="sample", annualize=True, periods_per_year=252)
    ref = returns_df.dropna().cov() * 252
    pd.testing.assert_frame_equal(Sigma, ref)


def test_covariance_sample_not_annualized(returns_df):
    Sigma = analysis.covariance(returns_df, method="sample", annualize=False)
    ref = returns_df.dropna().cov()
    pd.testing.assert_frame_equal(Sigma, ref)


def test_covariance_ledoit_wolf(returns_df):
    # Skip test if sklearn is not installed
    pytest.importorskip("sklearn")
    Sigma = analysis.covariance(returns_df, method="ledoit_wolf", annualize=True, periods_per_year=252)
    # Basic shape & index checks; LW result won't equal sample cov exactly
    assert isinstance(Sigma, pd.DataFrame)
    assert list(Sigma.index) == ["A", "B"]
    assert list(Sigma.columns) == ["A", "B"]
    assert (Sigma.values.diagonal() > 0).all()  # positive diagonals expected


def test_covariance_invalid_method(returns_df):
    with pytest.raises(ValueError):
        analysis.covariance(returns_df, method="unknown")


# ---------- portfolio_metrics ----------
def test_portfolio_metrics_basic(cleaned_returns):
    mu = cleaned_returns.mean() * 252
    Sigma = cleaned_returns.cov() * 252
    w = np.array([0.6, 0.4])

    out = analysis.portfolio_metrics(w, mu, Sigma, rf=0.0)

    ref_ret = float(np.dot(w, mu.values))
    ref_vol = float(np.sqrt(np.dot(w, Sigma.values @ w)))
    ref_sharpe = ref_ret / ref_vol if ref_vol > 0 else np.nan

    assert out["ret"] == pytest.approx(ref_ret)
    assert out["vol"] == pytest.approx(ref_vol)
    assert out["sharpe"] == pytest.approx(ref_sharpe)


def test_portfolio_metrics_weights_must_sum_to_one(cleaned_returns):
    mu = cleaned_returns.mean() * 252
    Sigma = cleaned_returns.cov() * 252
    w_bad = np.array([0.7, 0.4])  # sums to 1.1

    with pytest.raises(ValueError):
        analysis.portfolio_metrics(w_bad, mu, Sigma)


def test_portfolio_metrics_aligns_mu_to_Sigma_index(cleaned_returns):
    # Create mu with reversed order compared to Sigma
    mu = (cleaned_returns.mean() * 252).iloc[::-1]  # order B, A
    Sigma = cleaned_returns.cov() * 252            # index/cols A, B
    w = np.array([0.5, 0.5])

    out = analysis.portfolio_metrics(w, mu, Sigma)
    # Compute reference carefully using Sigma order (A,B)
    mu_aligned = mu.loc[Sigma.index]
    ref_ret = float(np.dot(w, mu_aligned.values))
    ref_vol = float(np.sqrt(np.dot(w, Sigma.values @ w)))

    assert out["ret"] == pytest.approx(ref_ret)
    assert out["vol"] == pytest.approx(ref_vol)


def test_portfolio_metrics_zero_vol_sharpe_nan():
    # Degenerate zero-covariance matrix
    mu = pd.Series({"A": 0.1, "B": 0.05})
    Sigma = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["A", "B"], columns=["A", "B"])
    w = np.array([0.5, 0.5])

    out = analysis.portfolio_metrics(w, mu, Sigma, rf=0.0)
    assert out["vol"] == 0.0
    assert np.isnan(out["sharpe"])


# ---------- _validate_returns behavior via public functions ----------
def test_public_funcs_dropna_via_validate(returns_df):
    # expected_returns should drop NaNs internally (via _validate_returns)
    mu = analysis.expected_returns(returns_df, method="mean", annualize=False)
    ref = returns_df.dropna().mean()
    pd.testing.assert_series_equal(mu, ref)

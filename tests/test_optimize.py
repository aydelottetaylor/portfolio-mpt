import numpy as np
import pandas as pd
import pytest

import portfolio_mpt.optimize as opt


# ---------- Fixtures ----------
@pytest.fixture
def returns_df():
    # Deterministic, small returns panel (no NaNs)
    # Three assets with different profiles
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.00, 0.015, 0.005, 0.01, 0.0, 0.02, -0.005],
            "B": [0.0, 0.01, 0.02, -0.015, 0.0, 0.01, 0.0, 0.005, 0.01, 0.0],
            "C": [-0.005, 0.0, 0.01, 0.0, 0.0, 0.015, -0.005, 0.02, 0.0, 0.01],
        },
        index=idx,
    )
    return df


@pytest.fixture
def mu_Sigma(returns_df):
    # Annualize using 252 trading days to match library expectations
    mu = returns_df.mean() * 252
    Sigma = returns_df.cov() * 252
    return mu, Sigma


# ---------- Helpers ----------
def _is_long_only(w, tol=1e-8):
    return (w >= -tol).all() and (w <= 1 + tol).all()


# ---------- Tests ----------
def test_gmv_constraints_and_properties(mu_Sigma):
    mu, Sigma = mu_Sigma
    res = opt.global_min_variance(mu, Sigma)

    # sums to 1, long-only bounds, success flag
    assert pytest.approx(res.weights.sum(), abs=1e-6) == 1.0
    assert _is_long_only(res.weights)
    assert res.success

    # GMV vol <= min single-asset vol
    single_vols = np.sqrt(np.diag(Sigma.values))
    assert res.vol <= single_vols.min() + 1e-8


def test_optimize_for_target_return_feasibility(mu_Sigma):
    mu, Sigma = mu_Sigma
    lo, hi = float(mu.min()), float(mu.max())

    with pytest.raises(ValueError):
        opt.optimize_for_target_return(mu, Sigma, target=lo - 1e-3)

    with pytest.raises(ValueError):
        opt.optimize_for_target_return(mu, Sigma, target=hi + 1e-3)


def test_optimize_for_target_return_hits_target(mu_Sigma):
    mu, Sigma = mu_Sigma
    lo, hi = float(mu.min()), float(mu.max())
    target = (lo + hi) / 2.0
    res = opt.optimize_for_target_return(mu, Sigma, target=target)

    # constraints
    assert pytest.approx(res.weights.sum(), abs=1e-6) == 1.0
    assert _is_long_only(res.weights)
    # target met (within optimization tolerance)
    assert res.ret == pytest.approx(target, rel=1e-4, abs=1e-6)
    assert res.success


def test_max_sharpe_beats_single_assets(mu_Sigma):
    mu, Sigma = mu_Sigma
    rf = 0.02
    res = opt.max_sharpe(mu, Sigma, rf=rf)

    # constraints
    assert pytest.approx(res.weights.sum(), abs=1e-6) == 1.0
    assert _is_long_only(res.weights)
    assert res.success

    single_vols = np.sqrt(np.diag(Sigma.values))
    single_sharpes = (mu.values - rf) / single_vols
    # Max Sharpe should be at least as good as the best single asset (allow tiny tolerance)
    assert res.sharpe >= single_sharpes.max() - 1e-4


def test_efficient_frontier_monotone_and_constraints(mu_Sigma):
    mu, Sigma = mu_Sigma
    lo, hi = float(mu.min()), float(mu.max())
    npts = 15

    front = opt.efficient_frontier(mu, Sigma, n_points=npts, ret_min=lo, ret_max=hi)
    assert len(front) == npts

    # Returns should be non-decreasing across the frontier targets
    rets = np.array([r.ret for r in front])
    assert np.all(np.diff(rets) >= -1e-6)

    # Vol should be (near) non-decreasing due to clipping in implementation
    vols = np.array([r.vol for r in front])
    assert np.all(np.diff(vols) >= -1e-6)

    # Each point respects constraints
    for r in front:
        assert pytest.approx(r.weights.sum(), abs=1e-6) == 1.0
        assert _is_long_only(r.weights)
        assert r.vol >= 0.0


def test_determinism_same_inputs_same_solution(mu_Sigma):
    mu, Sigma = mu_Sigma
    r1 = opt.global_min_variance(mu, Sigma)
    r2 = opt.global_min_variance(mu, Sigma)

    np.testing.assert_allclose(r1.weights, r2.weights, rtol=0, atol=1e-8)
    assert r1.ret == pytest.approx(r2.ret, abs=1e-10)
    assert r1.vol == pytest.approx(r2.vol, abs=1e-10)

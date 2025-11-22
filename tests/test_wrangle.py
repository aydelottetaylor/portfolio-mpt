import numpy as np
import pandas as pd
import pytest
import portfolio_mpt.wrangle as w


@pytest.fixture
def price_df():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = {
        "A": [100, 101, 103, 102, 104],
        "B": [50, 52, 51, 53, 54],
    }
    return pd.DataFrame(data, index=dates)


# ---------- to_returns ----------
def test_to_returns_simple(price_df):
    rets = w.to_returns(price_df, method="simple")
    expected = price_df.pct_change().dropna()
    pd.testing.assert_frame_equal(rets, expected)


def test_to_returns_log(price_df):
    rets = w.to_returns(price_df, method="log")
    expected = np.log(price_df / price_df.shift(1)).dropna()
    pd.testing.assert_frame_equal(rets, expected)


def test_to_returns_excess_with_benchmark_column(price_df):
    df = price_df.copy()
    df["BM"] = df["A"] * 0.9  # dummy benchmark
    rets = w.to_returns(df, method="excess", benchmark="BM")
    bench_rets = df["BM"].pct_change()
    expected = df.pct_change().sub(bench_rets, axis=0).dropna()
    pd.testing.assert_frame_equal(rets, expected)


def test_to_returns_excess_with_external_series(price_df):
    bench = price_df["A"] * 0.95
    rets = w.to_returns(price_df, method="excess", benchmark=bench)
    bench_rets = bench.pct_change()
    expected = price_df.pct_change().sub(bench_rets, axis=0).dropna()
    pd.testing.assert_frame_equal(rets, expected)


def test_to_returns_invalid_method(price_df):
    with pytest.raises(ValueError):
        w.to_returns(price_df, method="invalid")


def test_to_returns_excess_requires_benchmark(price_df):
    with pytest.raises(ValueError):
        w.to_returns(price_df, method="excess")


# ---------- cumulative_returns ----------
def test_cumulative_returns(price_df):
    rets = w.to_returns(price_df)
    cum = w.cumulative_returns(rets)
    expected = (1 + rets).cumprod() - 1
    pd.testing.assert_frame_equal(cum, expected)


# ---------- rolling_returns ----------
def test_rolling_returns_simple(price_df):
    rets = w.to_returns(price_df)
    roll = w.rolling_returns(rets, window=2)
    expected = (1 + rets).rolling(2).apply(lambda x: x.prod() - 1)
    pd.testing.assert_frame_equal(roll, expected)


# ---------- log_to_simple / simple_to_log ----------
def test_log_simple_conversions_roundtrip(price_df):
    rets = w.to_returns(price_df, method="simple")
    logr = w.simple_to_log(rets)
    simp = w.log_to_simple(logr)
    pd.testing.assert_frame_equal(simp, rets, check_exact=False, rtol=1e-10, atol=1e-10)


def test_simple_to_log_invalid_values():
    bad = pd.Series([-1.0, -1.1, 0.0])
    with pytest.raises(ValueError):
        w.simple_to_log(bad)

import json
import pandas as pd
import pytest
from unittest.mock import patch

import portfolio_mpt.data as data


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    # Redirect DATA_RAW to a temp directory for isolation
    d = tmp_path / "data_raw"
    monkeypatch.setattr(data, "DATA_RAW", d)
    d.mkdir()
    return d


@pytest.fixture
def dummy_spec():
    return data.FetchSpec(
        tickers=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-05",
        interval="1d",
    )


@pytest.fixture
def fake_prices():
    dates = pd.date_range("2024-01-01", periods=3)
    df = pd.DataFrame(
        {"AAPL": [150.0, 151.0, 152.0], "MSFT": [300.0, 302.0, 305.0]}, index=dates
    )
    return df


# --- Core tests ---

def test_fetch_prices_saves_and_loads(tmp_data_dir, dummy_spec, fake_prices):
    with patch("portfolio_mpt.data.yf.download") as mock_dl:
        mock_dl.return_value = pd.concat(
            {"Close": fake_prices}, axis=1
        )  # mimic yfinance multiindex

        out = data.fetch_prices(dummy_spec)
        assert isinstance(out, pd.DataFrame)
        assert set(out.columns) == {"AAPL", "MSFT"}

        # Parquet and manifest written
        pq = tmp_data_dir / f"AAPL-MSFT_{dummy_spec.start}_{dummy_spec.end}_{dummy_spec.interval}.parquet"
        mf = tmp_data_dir / f"AAPL-MSFT_{dummy_spec.start}_{dummy_spec.end}_{dummy_spec.interval}.manifest.json"
        assert pq.exists() and mf.exists()

        # Manifest content
        manifest = json.loads(mf.read_text())
        assert manifest["rows"] == 3
        assert manifest["cols"] == ["AAPL", "MSFT"]
        assert "pandas" in manifest["library_versions"]

        # Re-fetch uses cached parquet when force=False
        with patch("portfolio_mpt.data.yf.download") as mock_dl2:
            out2 = data.fetch_prices(dummy_spec, force=False)
            mock_dl2.assert_not_called()
        pd.testing.assert_frame_equal(out, out2)


def test_manifest_and_parquet_paths(dummy_spec):
    mf = data._manifest_path(dummy_spec)
    pq = data._parquet_path(dummy_spec)
    assert mf.name.endswith(".manifest.json")
    assert pq.name.endswith(".parquet")
    assert all(t in pq.name for t in dummy_spec.tickers)


def test_load_latest_for_returns_matching(tmp_data_dir, dummy_spec, fake_prices):
    pq = tmp_data_dir / f"AAPL-MSFT_{dummy_spec.start}_{dummy_spec.end}_{dummy_spec.interval}.parquet"
    fake_prices.to_parquet(pq)

    found = data.load_latest_for(["MSFT", "AAPL"])
    assert isinstance(found, pd.DataFrame)
    pd.testing.assert_frame_equal(found, fake_prices, check_freq=False)


def test_load_latest_for_returns_none_when_no_match(tmp_data_dir):
    # Create unrelated parquet
    p = tmp_data_dir / "GOOG.parquet"
    pd.DataFrame({"GOOG": [1, 2, 3]}).to_parquet(p)
    result = data.load_latest_for(["AAPL"])
    assert result is None


def test_fetch_prices_force_refetch(tmp_data_dir, dummy_spec, fake_prices):
    with patch("portfolio_mpt.data.yf.download") as mock_dl:
        mock_dl.return_value = pd.concat({"Close": fake_prices}, axis=1)
        # First call creates files
        data.fetch_prices(dummy_spec)
        # Second call with force=True should hit mock again
        data.fetch_prices(dummy_spec, force=True)
        assert mock_dl.call_count == 2

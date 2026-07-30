from pathlib import Path

import pandas as pd
import pytest

from src.sources import (
    SSIFastConnectAdapter, SourceConfigurationError, VietstockAdapter,
    _normalize_fdr_ohlc, _normalize_vnstock_ohlc, import_point_in_time_table,
    normalize_vietstock_ohlc,
)


def test_ssi_adapter_fails_closed_without_credentials(monkeypatch):
    monkeypatch.delenv("SSI_CONSUMER_ID", raising=False)
    monkeypatch.delenv("SSI_CONSUMER_SECRET", raising=False)
    with pytest.raises(SourceConfigurationError):
        SSIFastConnectAdapter()


def test_pit_import_requires_availability_contract(tmp_path: Path):
    source = tmp_path / "membership.csv"
    pd.DataFrame({"ticker": ["AAA"]}).to_csv(source, index=False)
    with pytest.raises(ValueError):
        import_point_in_time_table(
            source, tmp_path / "out.parquet",
            {"ticker", "effective_from", "effective_to", "available_at", "source"},
            "index_membership",
        )


def test_vietstock_token_and_normalization():
    html = (
        '<form id="__CHART_AjaxAntiForgeryForm">'
        '<input name=__RequestVerificationToken type=hidden value=test-token>'
        '</form>'
    )
    assert VietstockAdapter._anti_forgery_token(html) == "test-token"
    raw = pd.DataFrame([{
        "TradingDate": "/Date(1704067200000)/",
        "OpenPrice": 10,
        "HighestPrice": 12,
        "LowestPrice": 9,
        "ClosePrice": 11,
        "AdjustPrice": 10.5,
        "TotalVol": 100,
        "TotalVal": 1100,
    }])
    out = normalize_vietstock_ohlc(raw, "vcb", "https://example.test")
    assert out.loc[0, "ticker"] == "VCB"
    assert out.loc[0, "close"] == 11
    assert out.loc[0, "available_at"] > out.loc[0, "date"]


def test_vnstock_normalization_converts_thousand_vnd():
    raw = pd.DataFrame([{
        "time": "2020-01-02 07:00:00",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 100,
    }])
    out = _normalize_vnstock_ohlc(raw, "vcb", "https://example.test")
    assert out.loc[0, "ticker"] == "VCB"
    assert out.loc[0, "close"] == 10_500
    assert out.loc[0, "trading_value"] == 1_050_000


def test_fdr_normalization_shifts_utc_session_date():
    raw = pd.DataFrame({
        "Open": [10_000],
        "High": [11_000],
        "Low": [9_000],
        "Close": [10_500],
        "Adj Close": [10_250],
        "Volume": [100],
    }, index=pd.DatetimeIndex(["2020-01-01"], name="Date"))
    out = _normalize_fdr_ohlc(raw, "vcb", "https://example.test")
    assert out.loc[0, "date"] == pd.Timestamp("2020-01-02")
    assert out.loc[0, "adjusted_close"] == 10_250

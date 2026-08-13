from pathlib import Path

import pandas as pd
import pytest

from src.sources import (
    CafeFPublicHistoryAdapter, SSIFastConnectAdapter, SourceConfigurationError,
    TradingEconomicsAdapter, VietstockAdapter,
    crawl_cafef_standalone_workspace,
    _normalize_cafef_ohlc, _normalize_fdr_ohlc, _normalize_vnstock_ohlc,
    import_point_in_time_table,
    normalize_hose_security_master, normalize_trading_economics_ohlc,
    normalize_vietstock_ohlc,
)
from src.data_pipeline import Paths, generate_fixture


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


def test_fdr_normalization_rolls_friday_to_monday():
    raw = pd.DataFrame({
        "Open": [10_000], "High": [11_000], "Low": [9_000],
        "Close": [10_500], "Adj Close": [10_250], "Volume": [100],
    }, index=pd.DatetimeIndex(["2020-05-22"], name="Date"))
    out = _normalize_fdr_ohlc(raw, "vcb", "https://example.test")
    assert out.loc[0, "date"] == pd.Timestamp("2020-05-25")


def test_cafef_normalization_converts_reported_units():
    raw = pd.DataFrame([{
        "Ngay": "02/01/2020", "GiaMoCua": 10.0, "GiaCaoNhat": 11.0,
        "GiaThapNhat": 9.0, "GiaDongCua": 10.5, "GiaDieuChinh": 10.25,
        "KhoiLuongKhopLenh": 100, "GiaTriKhopLenh": 0.00105,
    }])
    out = _normalize_cafef_ohlc(raw, "vcb")
    assert out.loc[0, "ticker"] == "VCB"
    assert out.loc[0, "close"] == 10_500
    assert out.loc[0, "adjusted_close"] == 10_250
    assert out.loc[0, "trading_value"] == 1_050_000
    assert out.loc[0, "adjustment_policy"] == "unverified"


def test_cafef_standalone_workspace_is_isolated_and_uses_official_identity(
    tmp_path: Path, monkeypatch,
):
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    paths = Paths(tmp_path)
    generate_fixture(paths, "2020-01-01", "2020-12-31", tickers, 7)
    master_path = paths.normalized / "security_master.parquet"
    master = pd.read_parquet(master_path)
    master["history_method"] = "exchange_listing_history"
    master["data_class"] = "real"
    master["source"] = "official_exchange_test"
    master["source_url"] = "https://example.test/hose"
    master.to_parquet(master_path, index=False)

    def fake_daily(_self, symbol, _start, _end):
        return pd.DataFrame([{
            "Symbol": symbol, "Ngay": "02/01/2020", "GiaMoCua": 10.0,
            "GiaCaoNhat": 11.0, "GiaThapNhat": 9.0, "GiaDongCua": 10.5,
            "GiaDieuChinh": 10.25, "KhoiLuongKhopLenh": 100,
            "GiaTriKhopLenh": 0.00105,
        }])

    monkeypatch.setattr(CafeFPublicHistoryAdapter, "daily_ohlc", fake_daily)
    canonical_before = (paths.normalized / "prices.parquet").read_bytes()
    workspace, manifest = crawl_cafef_standalone_workspace(
        paths, "2020-01-01", "2020-12-31", tickers, max_workers=2,
    )
    standalone = pd.read_parquet(Paths(workspace).normalized / "prices.parquet")
    standalone_master = pd.read_parquet(Paths(workspace).normalized / "security_master.parquet")
    assert manifest["collected_count"] == 8
    assert set(standalone["source"]) == {"cafef_public_history"}
    assert set(standalone["security_id"]) == set(standalone_master["security_id"])
    assert (paths.normalized / "prices.parquet").read_bytes() == canonical_before


def test_trading_economics_fails_closed_without_api_key(monkeypatch):
    monkeypatch.delenv("TRADING_ECONOMICS_API_KEY", raising=False)
    with pytest.raises(SourceConfigurationError):
        TradingEconomicsAdapter()


def test_trading_economics_is_normalized_as_crosscheck_only():
    raw = pd.DataFrame([{
        "Symbol": "VCB:VN", "Date": "02/01/2024", "Open": 85000,
        "High": 87000, "Low": 84000, "Close": 86500,
    }])
    out = normalize_trading_economics_ohlc(raw, "VCB", "VCB:VN")
    assert out.loc[0, "security_id"] == "HOSE:VCB"
    assert out.loc[0, "data_class"] == "real_crosscheck"
    assert "volume" not in out.columns


def test_hose_official_master_uses_exchange_dates_and_isin_identity():
    current = pd.DataFrame([{
        "id": 10, "code": "AAA", "isin": "VN000000AAA4", "name": "AAA Co",
        "securitiesType": 1, "ftdate": 1_577_923_200, "regDate": 1_577_836_800,
        "listDate": -62_135_596_800, "acceptDate": 1_577_836_800,
        "bloomberg": "FIGI-AAA",
    }])
    detail = {
        "id": 20, "code": "BBB", "isin": "VN000000BBB1", "name": "BBB Co",
        "securityTypeId": 1, "ftdate": 1_420_070_400, "regDate": 1_419_984_000,
        "listDate": None, "acceptDate": 1_419_984_000, "bloomberg": "FIGI-BBB",
    }
    events = pd.DataFrame([{
        "securityId": 20, "code": "BBB", "isin": "VN000000BBB1",
        "cancelDate": 1_609_459_200,
    }])
    out = normalize_hose_security_master(
        current, [detail], events, "2026-01-01T00:00:00+00:00"
    ).set_index("ticker")
    assert out.loc["AAA", "security_id"] == "VN000000AAA4"
    assert out.loc["AAA", "history_method"] == "exchange_listing_history"
    assert out.loc["AAA", "listing_date"] == pd.Timestamp("2020-01-02")
    assert out.loc["BBB", "delisting_date"] == pd.Timestamp("2021-01-01")

import pandas as pd

from src.data_17_8 import (
    _archive_member_is_safe,
    _data_17_8_action_scope,
    _metric_from_page,
    _parse_accounting_number,
    _valid_vnstock_checkpoint,
    classify_disclosure_title,
)
from src.data_pipeline import Paths


def test_disclosure_classifier_accepts_direct_financial_statement_only():
    row = classify_disclosure_title(
        "VCB: Báo cáo tài chính hợp nhất quý 2 năm 2024 đã soát xét"
    )
    assert row is not None
    assert row["ticker"] == "VCB"
    assert row["document_type"] == "financial_statement"
    assert row["fiscal_period_end"] == pd.Timestamp("2024-06-30")
    assert row["statement_scope"] == "consolidated"
    assert row["assurance"] == "reviewed"


def test_disclosure_classifier_rejects_reminder_that_mentions_bctc():
    assert classify_disclosure_title(
        "FIR: Nhắc nhở chậm CBTT BCTC kiểm toán năm 2025"
    ) is None


def test_disclosure_classifier_handles_annual_report():
    row = classify_disclosure_title("FPT: Báo cáo thường niên năm 2023")
    assert row is not None
    assert row["document_type"] == "annual_report"
    assert row["period_type"] == "annual"
    assert row["fiscal_period_end"] == pd.Timestamp("2023-12-31")


def test_vietnamese_accounting_number_parser():
    assert _parse_accounting_number("1.234.567") == 1_234_567
    assert _parse_accounting_number("(1.234.567)") == -1_234_567
    assert _parse_accounting_number("1,234,567") == 1_234_567


def test_metric_extraction_keeps_page_evidence_and_unit():
    page = {
        "page": 12,
        "method": "native",
        "text": "Đơn vị tính: triệu đồng\nTỔNG CỘNG TÀI SẢN 270 1.234.567 1.100.000",
    }
    fact = _metric_from_page(page, "total_assets")
    assert fact is not None
    assert fact["value"] == 1_234_567 * 1_000_000
    assert fact["page"] == 12
    assert fact["confidence"] == 0.90


def test_complete_case_scope_excludes_entire_ticker_with_unresolved_event(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "security_id": ["HOSE:AAA", "HOSE:BBB"],
    }).to_parquet(paths.normalized / "security_master.parquet", index=False)
    pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "event_type": ["CASH_DIVIDEND", "CASH_DIVIDEND"],
        "effective_date": ["2024-05-20", "2024-05-20"],
        "record_date": ["2024-05-21", "2024-05-21"],
        "announcement_date": ["2024-05-01", "2024-05-01"],
        "available_at": ["2024-05-01", "2024-05-01"],
        "verification_status": ["verified_cross_source", "conflict"],
    }).to_parquet(paths.normalized / "corporate_actions.parquet", index=False)

    eligible, excluded, complete = _data_17_8_action_scope(paths)

    assert set(eligible["ticker"]) == {"AAA"}
    assert excluded == {"BBB"}
    assert complete == {"AAA"}


def test_vnstock_checkpoint_requires_coverage_and_valid_ohlc(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=60)
    path = tmp_path / "AAA.parquet"
    pd.DataFrame({
        "date": dates,
        "ticker": "AAA",
        "open": 10_000.0,
        "high": 10_500.0,
        "low": 9_500.0,
        "close": 10_200.0,
        "volume": 100_000,
    }).to_parquet(path, index=False)

    frame = _valid_vnstock_checkpoint(
        path,
        ticker="AAA",
        expected_start=pd.Timestamp("2024-01-01"),
        expected_end=pd.Timestamp("2024-03-22"),
    )

    assert frame is not None
    assert len(frame) == 60


def test_archive_member_path_validation_rejects_traversal():
    assert _archive_member_is_safe("reports/annual_statement.pdf")
    assert not _archive_member_is_safe("../outside.pdf")
    assert not _archive_member_is_safe("C:/outside.pdf")

import pandas as pd

from src.corporate_actions import (
    CafeFCorporateActionAdapter,
    VSDCCorporateActionAdapter,
    reconcile_corporate_actions,
)
from src.price_adjustment import corporate_action_total_return


def test_cafef_parser_keeps_ex_date_and_separates_cash_and_stock_events():
    markup = """
    <a>Lịch sử trả cổ tức chia thưởng và tăng vốn »</a>
    <div class="tooltip"><div class="top"></div>
    <div class="middle" style="padding-left: 10px">
      - <b>22/12/2021</b>: Cổ tức bằng Cổ phiếu, tỷ lệ 27,6%<br />
      Cổ tức bằng Tiền, tỷ lệ 12%<br />
    </div></div>
    """
    rows = CafeFCorporateActionAdapter.parse(
        markup, "VCB", "https://example.test/vcb", "2026-08-15T00:00:00+00:00"
    )
    assert [row["event_type"] for row in rows] == ["STOCK_DIVIDEND", "CASH_DIVIDEND"]
    assert rows[0]["stock_dividend_ratio"] == 0.276
    assert rows[1]["cash_dividend_per_share"] == 1200.0
    assert rows[1]["ex_date"] == pd.Timestamp("2021-12-22")


def test_vsdc_parser_extracts_official_cash_terms_and_historical_availability():
    markup = """
    <main><h3>VCB: Chi trả cổ tức năm 2024 bằng tiền</h3>
    <div>Cập nhật ngày 26/09/2025 - 10:38:28</div>
    <p>Mã chứng khoán:</p><p>VCB</p><p>Mã ISIN:</p><p>VN000000VCB4</p>
    <p>Sàn giao dịch:</p><p>HOSE</p>
    <p>Ngày đăng ký cuối cùng:</p><p>06/10/2025</p>
    <p>Tỷ lệ thực hiện: 4,5%/cổ phiếu (01 cổ phiếu được nhận 450 đồng)</p>
    <p>Ngày thanh toán: 24/10/2025</p></main><h4>Tin cùng tổ chức</h4>
    """
    rows = VSDCCorporateActionAdapter.parse(
        markup, "https://example.test/notice", "2026-08-15T00:00:00+00:00"
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["security_id"] == "VN000000VCB4"
    assert row["event_type"] == "CASH_DIVIDEND"
    assert row["cash_dividend_per_share"] == 450.0
    assert row["record_date"] == pd.Timestamp("2025-10-06")
    assert row["payment_date"] == pd.Timestamp("2025-10-24")
    assert row["available_at"] == pd.Timestamp("2025-09-26 10:38:28")


def test_cross_source_reconciliation_verifies_only_compatible_event_terms():
    official = pd.DataFrame([{
        "security_id": "VN000000VCB4", "ticker": "VCB", "event_type": "CASH_DIVIDEND",
        "announcement_date": "2025-09-26", "record_date": "2025-10-06",
        "cash_dividend_per_share": 450.0, "source": "vsdc_official_notice",
        "source_url": "https://example.test/official", "available_at": "2025-09-26",
        "fetched_at": "2026-08-15", "raw_checksum": "a", "parser_version": "test",
    }])
    reference = pd.DataFrame([{
        "ticker": "VCB", "event_type": "CASH_DIVIDEND", "ex_date": "2025-10-03",
        "effective_date": "2025-10-03", "cash_dividend_per_share": 450.0,
        "source": "cafef_public_corporate_history", "source_url": "https://example.test/ref",
        "available_at": "2026-08-15", "fetched_at": "2026-08-15",
        "raw_checksum": "b", "parser_version": "test",
    }])
    master = pd.DataFrame([{
        "security_id": "VN000000VCB4", "ticker": "VCB", "listing_date": "2009-06-30",
        "delisting_date": None,
    }])
    ledger, conflicts = reconcile_corporate_actions(official, reference, master)
    assert conflicts.empty
    row = ledger.iloc[0]
    assert row["verification_status"] == "verified_cross_source"
    assert row["ex_date"] == pd.Timestamp("2025-10-03")


def test_total_return_cash_dividend_fixture():
    assert corporate_action_total_return(
        10_000, 9_000, cash_dividend_per_share=1_000,
    ) == 0.0


def test_total_return_stock_dividend_and_bonus_fixture():
    actual = corporate_action_total_return(
        10_000, 8_000, stock_dividend_ratio=0.20, bonus_share_ratio=0.05,
    )
    assert actual == 0.0


def test_total_return_split_and_reverse_split_have_distinct_ratios():
    assert corporate_action_total_return(10_000, 5_000, split_ratio=2.0) == 0.0
    assert corporate_action_total_return(10_000, 20_000, reverse_split_ratio=2.0) == 0.0


def test_total_return_rights_issue_deducts_subscription_cash():
    # One old share at 10,000 grants 0.5 new share at 4,000. At an ex-price of
    # 8,000, terminal wealth is 8,000*1.5 - 4,000*0.5 = 10,000.
    assert corporate_action_total_return(
        10_000, 8_000, rights_ratio=0.5, rights_subscription_price=4_000,
    ) == 0.0

# BÁO CÁO VẬN HÀNH HỆ THỐNG RESEARCH V2

| Trường | Giá trị |
|---|---|
| Experiment ID | `BLOCKED_DATA_GATES` |
| Dataset hash (candidate) | `dd24d243f3e00d6516962326474ff243cdc200e9667dd1b4a7dd5cd3ed664b82` |
| Adjustment version | `corporate-actions-v2` |
| Config hash | `not_created_no_confirmatory_run` |
| Git commit tại thời điểm báo cáo | `29c9d216e4c5f0539fd7fc5a8d26d63d1e4510b8` |
| Created at | `2026-08-15T06:06:22.718266+00:00` |
| Mode | `research_v2_fail_closed` |
| Label | `BLOCKED — DIAGNOSTIC ARTIFACT, NOT CONFIRMATORY RESEARCH` |
| Folds | `0 (không huấn luyện/backtest khi gate chưa đạt)` |
| OOS start/end | `not_created` |
| Audit status | `blocked_valid` |

## 1. Phạm vi và nguyên tắc diễn giải

Hệ thống đã hoàn tất các phần có thể thực hiện độc lập gồm thu thập và hòa giải sự kiện doanh nghiệp, xây dựng bộ giá ứng viên, kiểm toán điều chỉnh giá, xây dựng universe point-in-time và đối chứng return-only. Pipeline xác nhận không được chạy vì cổng dữ liệu chưa đạt. Đây là hành vi fail-closed theo thiết kế, không phải một kết quả backtest thất bại.

## 2. Nguồn dữ liệu

- **HOSE official website/API**: usable_now=`True`; vai trò: security identity and listing/delisting history; giới hạn: historical EOD OHLC and total-return index feeds are licensed services.
- **VSDC public notices**: usable_now=`True`; vai trò: corporate-action evidence; giới hạn: official event terms are parsed, but confirmatory use still requires independent ex-date corroboration and zero unresolved material events.
- **Vietstock Finance**: usable_now=`False`; vai trò: authenticated OHLCV cross-check; giới hạn: no reusable cookie/header file configured; no credential is guessed or persisted.
- **SSI FastConnect**: usable_now=`False`; vai trò: official broker OHLC and index components; giới hạn: consumer ID/secret absent.
- **Trading Economics**: usable_now=`False`; vai trò: OHLC cross-check and macro snapshot; giới hạn: API key absent; market history lacks volume and verified adjustment semantics.
- **World Bank Indicators API v2**: usable_now=`True`; vai trò: Vietnam macro snapshot; giới hạn: current revised snapshot has no observation-specific release vintage.
- **FinanceDataReader/Yahoo**: usable_now=`True`; vai trò: historical OHLCV checkpoint source; giới hạn: not exchange-official; adjustment policy remains uncertified.
- **vnstock/KBS**: usable_now=`True`; vai trò: OHLCV fallback and cross-check; giới hạn: not exchange-official; delisted coverage and adjustment semantics are incomplete.
- **CafeF public history**: usable_now=`True`; vai trò: last-resort historical OHLCV gap coverage; giới hạn: aggregated reference data; not exchange-official and adjustment semantics remain uncertified.
- **CafeF corporate-action history**: usable_now=`True`; vai trò: independent ex-date corroboration; giới hạn: used to corroborate VSDC events, not treated as an exchange-official source or as a standalone adjustment authority.
- **IMF SDMX**: usable_now=`False`; vai trò: international macroeconomic series; giới hạn: not integrated because a release-vintage contract for the selected Vietnam series has not been verified.
- **Vietnam National Statistics Office PX-Web**: usable_now=`False`; vai trò: official domestic macro statistics; giới hạn: no stable dataset/API and historical publication-time contract selected for this study.
- **State Bank of Vietnam**: usable_now=`False`; vai trò: policy rates, exchange rates and banking statistics; giới hạn: no stable bulk API/release-vintage adapter verified in this repository.

## 3. Corporate actions

VSDC cung cấp 2,261 bản ghi sự kiện chính thức và CafeF cung cấp 2,359 bản ghi đối chiếu. Ledger sau hòa giải có 2,655 dòng, trong đó 1,698 dòng được xác minh chéo; 302 dòng chưa xác minh ex-date; 250 dòng xung đột và 405 dòng chỉ có nguồn tham khảo.

## 4. Kiểm toán điều chỉnh giá

Price-adjustment gate có trạng thái **blocked**. Hệ thống đã áp dụng 1,698 sự kiện đã xác minh vào bộ dữ liệu ứng viên, nhưng vẫn còn 956 sự kiện trọng yếu chưa giải quyết, 1,531 thay đổi source-adjusted trên 1 điểm phần trăm chưa ghép được với sự kiện và 2,311 biến động raw vượt biên độ HOSE đã ghép theo khoảng cách phiên mà chưa có sự kiện giải thích. Vì vậy `research_eligible=false`.

## 5. Đối chứng trực tiếp trên danh mục Data A đã đóng băng

| Định nghĩa lợi nhuận | Gross cumulative return | Net cumulative return |
|---|---:|---:|
| Raw close | -17.91% | -21.77% |
| Source-adjusted close | -16.78% | -20.69% |
| Research total-return candidate | -13.55% | -17.62% |

Phép so sánh này giữ nguyên mã, tỷ trọng và ngày tái cân bằng; do đó chỉ định lượng tác động trực tiếp của định nghĩa lợi nhuận. Cột research total-return vẫn là candidate vì gate bị chặn. Chênh lệch này không chứng minh corporate actions là nguyên nhân của lợi nhuận âm.

## 6. Historical universe point-in-time

Security master PIT chứa 500 mã trên 72 ngày quyết định tháng, giữ lại 96 mã đã hủy niêm yết và không dùng bộ lọc độ đầy đủ toàn giai đoạn tương lai. Trạng thái audit là `partial_blocked` vì chưa có lịch sử đình chỉ đầy đủ và chưa xác minh toàn bộ đổi mã/sáp nhập pháp nhân.

## 7. Baseline Data A được bảo toàn

Data A có dataset hash `6e046b509fef366681866328d5bd99ec63541c2de8597f0e7bebc101813baa05`, 12/12 folds, Rank IC trung bình 0.079541, cumulative net return -20.69%, Sharpe -0.669293, maximum drawdown -35.00%. XY-QAOA có feasibility 100.00%, primary gap trung bình 8.93% và best-observed gap trung bình 0.00%. Danh mục fold cuối: SMB, DXG, GAS, STB.

## 8. H1–H6

- **H1**: `not_statistically_supported` (Data A). Research V2: `not_testable_due_to_data`.
- **H2**: `not_statistically_supported` (Data A). Research V2: `not_testable_due_to_data`.
- **H3**: `statistically_supported_on_declared_tests` (Data A). Research V2: `not_testable_due_to_data`.
- **H4**: `not_statistically_supported` (Data A). Research V2: `not_testable_due_to_data`.
- **H5**: `not_statistically_supported` (Data A). Research V2: `not_testable_due_to_data`.
- **H6**: `sensitivity_completed` (Data A). Research V2: `not_testable_due_to_data`.

Research V2 không kế thừa trạng thái giả thuyết từ Data A. Khi chưa có experiment xác nhận, cả H1–H6 đều mang trạng thái `not_testable_due_to_data`.

## 9. Kết luận

Code path, dữ liệu chẩn đoán và blocked artifact đã được tạo có kiểm toán. Không có experiment ID, config hash, danh mục fold cuối hay kết quả H1–H6 mới vì hệ thống đã chủ động không huấn luyện trên dữ liệu chưa đạt hợp đồng xác nhận.

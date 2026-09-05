# Data 5/9 — Bộ dữ liệu nộp bài NCKH

Đây là bản đóng gói có cấu trúc của **đúng bộ dữ liệu đã chạy mô hình cuối cùng** trong báo cáo. Không có ô dữ liệu nào bị sửa, bổ sung hoặc nội suy khi đổi tên gói.

## Tệp chính

- `data_5_9.csv`: dữ liệu đầy đủ, 179,173 bản ghi × 64 cột.
- `data_5_9.zip`: bản nén để tải lên Colab; chỉ chứa `data_5_9.csv`.
- `schema.json`: định nghĩa 64 cột và các trường bắt buộc cho Colab.
- `manifest_5_9.json`: nguồn gốc, hash, thống kê và kết quả kiểm định máy.
- `AUDIT_DATA_5_9.md`: biên bản audit dễ đọc.
- `evidence/`: manifest lần chạy cuối và các bảng kết quả đối chiếu báo cáo.

## Bằng chứng đúng dữ liệu chạy mô hình

SHA-256 của CSV này là `b0a16d9f8c31a2a5d4e1ba8f00d49b50f112f149d4fae23b3529df085a45ccb2`. Hash đó trùng với `dataset_sha256` trong `evidence/run_manifest.json`, nơi mô hình cuối khai báo đầu vào `data_29_8.csv`.

## Phạm vi dữ liệu

- PRICE: 174,626 dòng, 120 mã, từ 2020-01-02 đến 2026-08-28.
- Tổng cộng: 179,173 dòng gồm PRICE, BENCHMARK, CORPORATE_ACTION, SECURITY và METADATA.
- Dữ liệu đến 2025-12-31: `historical_audited_research_panel`.
- Phần mở rộng 2026 đến 2026-08-28: `provisional_observed_extension` (16,800 dòng, 105 mã).

## Cách dùng

Giải nén `data_5_9.zip` hoặc đọc trực tiếp `data_5_9.csv`. Pipeline cần sáu trường tối thiểu: `record_type`, `date`, `ticker`, `adjusted_close`, `volume`, `trading_value`.

## Giới hạn sử dụng

Gói này phục vụ nghiên cứu và tái lập kết quả báo cáo; không cho phép suy diễn rằng vốn thật đã được triển khai hoặc lợi thế lượng tử đã được chứng minh.

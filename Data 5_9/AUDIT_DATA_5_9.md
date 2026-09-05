# Biên bản audit dữ liệu Data 5/9

**Kết luận: PASS.** `data_5_9.csv` là bản sao byte-for-byte của dữ liệu được khai báo trong lần chạy mô hình cuối.

| Kiểm tra | Kết quả |
|---|---|
| source_sha_matches_final_run_manifest | PASS |
| copied_csv_is_byte_identical | PASS |
| row_count_is_179173 | PASS |
| column_count_is_64 | PASS |
| required_columns_present | PASS |
| record_counts_match_report | PASS |
| price_ticker_count_is_120 | PASS |
| price_period_matches_report | PASS |
| provisional_2026_extension_matches_manifest | PASS |
| no_duplicate_price_ticker_dates | PASS |
| no_missing_adjusted_close | PASS |
| positive_adjusted_close | PASS |
| nonnegative_volume | PASS |
| nonnegative_trading_value | PASS |
| schema_required_columns_match | PASS |
| zip_contains_only_data_5_9_csv | PASS |
| zip_member_is_byte_identical | PASS |
| zip_crc_integrity | PASS |

## Đối chiếu định lượng

| Chỉ tiêu | Kết quả audit |
|---|---:|
| Tổng số dòng | 179,173 |
| Tổng số cột | 64 |
| Dòng PRICE | 174,626 |
| Số mã PRICE | 120 |
| Ngày đầu | 2020-01-02 |
| Ngày cuối | 2026-08-28 |
| Dòng mở rộng 2026 | 16,800 |
| Mã có dữ liệu mở rộng 2026 | 105 |
| PRICE trùng ticker-date | 0 |
| PRICE thiếu adjusted_close | 0 |

## Chuỗi bằng chứng

1. Dữ liệu nguồn: `data_29_8.csv`.
2. Hash nguồn: `b0a16d9f8c31a2a5d4e1ba8f00d49b50f112f149d4fae23b3529df085a45ccb2`.
3. Manifest lần chạy cuối khai báo đúng hash trên.
4. Hash `data_5_9.csv`: `b0a16d9f8c31a2a5d4e1ba8f00d49b50f112f149d4fae23b3529df085a45ccb2` — trùng hoàn toàn.
5. CSV trong ZIP có cùng hash và ZIP vượt kiểm tra CRC.

Các bảng trong `evidence/` là bản sao từ thư mục kết quả cuối để người chấm có thể đối chiếu giả thuyết, kết quả thực nghiệm và danh mục tháng 9/2026 với báo cáo.

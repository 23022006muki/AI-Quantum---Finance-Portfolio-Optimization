# Research data remediation status

Ngày kiểm tra: 2026-08-06

Nhánh: `codex/rebuild-pit-research-data`

| Hạng mục | Trước thay đổi | Trạng thái hiện tại | Điều kiện để đóng |
|---|---|---|---|
| Price identity | Chỉ có ticker | 466.701 dòng được gắn ISIN từ HOSE; 463 dòng SHB trước ngày chuyển sàn được cách ly có thể phục hồi | Đã đóng cho panel hiện tại |
| Universe | 300 mã từ current listing | Security master chính thức có 404 mã hiện hành + 96 mã hủy niêm yết; top 300 động theo trailing liquidity | Cần OHLC cho 145/445 mã liên quan còn thiếu trước khi research run |
| Giá và staging | Adapter có thể ghi thẳng normalized | Panel mới được validate trong staging, panel cũ được archive rồi mới promote nguyên tử | Chạy lại bằng nguồn được cấp quyền |
| SSI/Vietstock | Adapter cơ bản | Có checkpoint, retry, raw content-addressed archive và credential file | Cần credential mới của người dùng |
| Trading Economics | Chưa có | Adapter API chính thức, chỉ dùng cross-check OHLC | Cần `TRADING_ECONOMICS_API_KEY` |
| HOSE listing history | Dùng ngày giá đầu tiên làm proxy | Đã crawl và promote 500 bản ghi với ISIN, ngày giao dịch đầu tiên và ngày hủy niêm yết chính thức | Đã đóng cho giai đoạn 2015–2025 |
| Outlier | 46 dòng unresolved sau khi loại dữ liệu ngoài khoảng HOSE | Có reconciliation ledger và đối chiếu corporate action/cross-source | Cần corporate actions hoặc nguồn xác minh từng dòng |
| Price adjustment | Chưa xác minh | SHA-bound contract vẫn fail-closed | Cần phương pháp điều chỉnh do provider công bố/người có thẩm quyền xác nhận |
| Benchmark | Chưa có | Import từ chối price index giả total-return | Cần total-return series và methodology URL |
| Research run | `20260806T210525-72a01af202-blocked` được audit `blocked_valid` | Bị chặn trước train/backtest; không sinh metrics | Bổ sung OHLC còn thiếu, xử lý 46 outlier, chứng nhận adjustment và total-return benchmark |

Không hạng mục nào được chuyển sang “đã xác minh” chỉ vì code có khả năng xử lý nó.

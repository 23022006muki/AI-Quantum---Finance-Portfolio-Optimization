# Run report

## Trạng thái triển khai

Nhánh `fix/research-validity` đã hoàn thiện luồng nghiên cứu AI–Quantum theo hướng
fail-closed. QUBO dùng vector lợi nhuận kỳ vọng được hiệu chỉnh từ XGBoost trên validation
đã purge; EWMA đa biến ước lượng hiệp phương sai và làm đối chứng. Universe toàn HOSE và
universe theo thành phần chỉ số được khai báo tách biệt.

Backtest hiện xử lý buy-and-hold drift, missing return, hủy niêm yết có xác minh, giới hạn
tỷ trọng/ngành/turnover/ADV, đồng thời tách commission, thuế bán, slippage và market impact.
Benchmark total-return và risk-free PIT có hợp đồng import riêng. H1–H6, ablation,
sensitivity đa seed, noise stress, bootstrap–Holm và artifact SHA-256 đều có đầu ra kiểm toán.

## Kiểm thử phần mềm

- `42 passed`.
- Có research-mode integration test thành công với hợp đồng dữ liệu tổng hợp hợp lệ.
- Audit phát hiện artifact bị sửa sau khi chạy.
- Demo fixture gần nhất audit `pass`; fixture không phải kết quả nghiên cứu.

## Research gate trên panel hiện có

Panel thật có 467.164 dòng, 300 mã, từ 2020-01-02 đến 2025-12-31. Lần chạy
`20260806T181627-c5ef044e1b-blocked` được audit là `blocked_valid` và dừng trước model vì:

1. data-quality còn 47 adjusted-return outliers chưa xác minh;
2. lịch sử niêm yết/hủy niêm yết vẫn dùng `first_price_observation_proxy`;
3. universe snapshot vì thế chưa có nguồn lịch sử đủ tin cậy;
4. chưa có hợp đồng điều chỉnh giá đã xác minh;
5. chưa có VN-Index total-return point-in-time theo data contract.

Các bảng phụ fixture cũ đã được chuyển có thể phục hồi sang
`outputs/quarantine/fixture_auxiliary/20260806T180922`. Không có metrics H1–H6 hoặc tuyên
bố quantum advantage được phát hành từ run bị chặn.

## Lệnh kiểm tra

```powershell
python -m pytest -q
python -m compileall -q src app.py scripts tests
python -m src.cli run-full --config configs/hose300_real.yaml
python scripts/audit_research_run.py outputs/experiments/20260806T181627-c5ef044e1b-blocked --allow-blocked
```

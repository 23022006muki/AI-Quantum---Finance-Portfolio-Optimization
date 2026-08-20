# Báo cáo kết quả hệ thống Data 17/8

**Ngày hoàn tất:** 20/08/2026  
**Experiment ID:** `20260820T160429-4f2cfc123d`  
**Dataset SHA-256:** `69d93537d08d09d958fc80987e1297c05d3dd75bb4d40b6685bcbfbadf50892b`  
**Trạng thái:** Chạy thành công 33/33 fold; kết quả mang tính khám phá.

## 1. Dữ liệu và thiết kế thực nghiệm

Hệ thống sử dụng 157.826 quan sát của 120 cổ phiếu trong giai đoạn 02/01/2020–31/12/2025. Giá thô và giá điều chỉnh được kiểm tra chéo giữa CafeF và KBS; benchmark là VNAllShare TRI. Dữ liệu đạt kiểm tra chất lượng, còn leakage audit đạt mức `pass_with_limitations` do nghiên cứu chưa sử dụng dữ liệu báo cáo tài chính point-in-time, dữ liệu vĩ mô và giao dịch nước ngoài trong mô hình chính.

Thực nghiệm sử dụng walk-forward backtest 33 fold, tái cân bằng hàng tháng, XGBoost để xếp hạng, EWMA để ước lượng hiệp phương sai, adaptive universe reduction, QUBO chọn 4 tài sản và XY-QAOA trên ideal statevector simulator. Tỷ trọng được tối ưu cổ điển, có xét giới hạn tỷ trọng, lô giao dịch, sức chứa ADV và chi phí giao dịch. Quy mô danh mục cơ sở là 50 triệu đồng; 100 triệu đồng và 1 tỷ đồng chỉ là kịch bản kiểm tra sức chứa.

## 2. Kết quả tín hiệu và solver

Rank IC trung bình của XGBoost là 0,0292 và trung vị là 0,0388. Chênh lệch Rank IC giữa XGBoost và EWMA không có ý nghĩa thống kê sau hiệu chỉnh Holm.

| Bộ giải | Feasibility rate | Optimality gap trung bình | Thời gian trung bình |
|---|---:|---:|---:|
| Exact | 100,00% | 0,0000 | 0,0000 giây |
| Simulated Annealing | 100,00% | 0,0000 | 0,0319 giây |
| Penalty-QAOA | 58,51% | 0,1741 | 0,3039 giây |
| XY-QAOA với Dicke state | 100,00% | 0,0269 | 0,1609 giây |

XY-QAOA duy trì đúng ràng buộc cardinality trong toàn bộ các lần chạy chính và có optimality gap thấp hơn Penalty-QAOA. Kết quả không được diễn giải là bằng chứng về quantum advantage vì thuật toán chạy trên simulator lý tưởng.

## 3. Hiệu quả danh mục ngoài mẫu

| Chiến lược | Lợi nhuận tích lũy | Lợi nhuận năm hóa | Biến động năm hóa | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| Full pipeline XY-QAOA | **2,25%** | 0,62% | 5,89% | -0,3673 | -13,50% |
| VNAllShare TRI | 43,75% | 10,61% | 21,48% | 0,4400 | -37,94% |
| Liquidity Top-K + Exact | 38,65% | 9,50% | 32,34% | 0,3518 | -43,78% |
| Minimum Variance | 11,71% | 3,12% | 11,79% | 0,0693 | -17,49% |
| Equal Weight Universe | -7,54% | -2,15% | 18,23% | -0,1899 | -44,96% |
| EWMA Top-K + Exact | -19,48% | -5,84% | 34,42% | -0,0883 | -50,74% |
| Adaptive Exact | -34,71% | -11,17% | 22,20% | -0,5543 | -51,18% |

Pipeline tạo lợi nhuận dương và kiểm soát drawdown tốt hơn benchmark, nhưng không vượt benchmark về lợi nhuận hoặc hiệu quả điều chỉnh theo rủi ro. Sharpe âm vì lợi nhuận năm hóa 0,62% thấp hơn lãi suất phi rủi ro giả định 3%/năm. Các kiểm định bootstrap không ghi nhận chênh lệch lợi nhuận có ý nghĩa thống kê giữa pipeline đầy đủ và các chiến lược đối chứng.

## 4. Danh mục ở fold cuối

Ngày quyết định cuối là 30/12/2024, trong trạng thái thị trường `risk_off`. Mức phân bổ cổ phiếu mục tiêu là 25%; sau làm tròn lô, tỷ trọng cổ phiếu thực thi là 18,73% và tiền mặt là 81,27%.

| Mã | Tỷ trọng thực thi |
|---|---:|
| PGC | 6,28% |
| VPD | 5,38% |
| PAN | 4,77% |
| TDC | 2,30% |
| Tiền mặt | 81,27% |

## 5. Đánh giá giả thuyết

- H1 không được hỗ trợ thống kê: XGBoost chưa chứng minh Rank IC tốt hơn EWMA.
- H2 chỉ được hỗ trợ ở khía cạnh đa dạng hóa: adaptive universe reduction giảm tương quan so với Top-M, nhưng chưa cải thiện forward return có ý nghĩa thống kê.
- H3 được hỗ trợ: XY-QAOA có feasibility rate cao hơn Penalty-QAOA.
- H4 được hỗ trợ khi so sánh với Penalty-QAOA: XY-QAOA có optimality gap thấp hơn; tuy nhiên vẫn kém exact solver và simulated annealing trong cấu hình này.
- H5 không được hỗ trợ: pipeline đầy đủ chưa tạo hiệu quả tài chính vượt trội có ý nghĩa thống kê.
- H6 đã hoàn thành phân tích độ nhạy; kết quả thay đổi theo độ sâu, số shots và cardinality nên không thể suy rộng ngoài lưới tham số đã khai báo.

## 6. Hạn chế và kết luận

Kho tài liệu doanh nghiệp đã lập chỉ mục BCTC và BCTN cho toàn bộ 394 mã trong security master, nhưng còn 9/3.094 tệp chuẩn hóa không tải được; dữ liệu tài chính point-in-time và ngành lịch sử vì vậy chưa được đưa vào mô hình chính. Đây là lý do run được phân loại là exploratory thay vì confirmatory.

Kết quả mới thay thế toàn bộ số liệu của các lần chạy cũ. Hệ thống hiện chứng minh được khả năng thực thi pipeline end-to-end, bảo toàn ràng buộc cardinality và tạo lợi nhuận dương nhẹ với drawdown thấp; chưa chứng minh được ưu thế tài chính hay quantum advantage.

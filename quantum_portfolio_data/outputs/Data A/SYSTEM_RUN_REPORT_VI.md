# Báo cáo vận hành hệ thống trên Data A

## 1. Dữ liệu sử dụng

Data A là bộ dữ liệu complete-case được tạo từ panel giá HOSE thật trong giai đoạn
02/01/2020–31/12/2025. Hệ thống giữ lại 565.471 bản ghi của 394 mã và loại 51 mã
không đạt điều kiện. Mỗi mã được giữ phải có ít nhất 40 quan sát hoàn chỉnh và không
có khoảng trống dẫn đầu, cuối kỳ hoặc nội kỳ dài quá 5 phiên. Hash SHA-256 của panel
giá là `6e046b509fef366681866328d5bd99ec63541c2de8597f0e7bebc101813baa05`.

Data-quality đạt `PASS`, không còn lỗi schema, OHLC, timestamp khả dụng, giá trị âm,
trùng khóa hay outlier lợi nhuận chưa xử lý. Data A vẫn là dữ liệu khám phá vì chưa có
hợp đồng xác minh điều chỉnh corporate actions và benchmark total-return.

## 2. Quy trình hệ thống

Hệ thống giới hạn universe động tối đa 300 mã tại mỗi kỳ tái cân bằng. Các đặc trưng
giá, động lượng, biến động, downside risk và thanh khoản được xây dựng riêng theo từng
cửa sổ thời gian. XGBoost tạo tín hiệu xếp hạng; EWMA được dùng cho tín hiệu đối chứng
và ước lượng ma trận hiệp phương sai đa biến.

Adaptive universe reduction kết hợp tín hiệu, thanh khoản, rủi ro và tương quan để
giảm universe xuống 8 ứng viên. Bài toán QUBO chọn đúng 4 mã và được giải bằng exact
solver, simulated annealing, penalty-QAOA và feasible-subspace XY-QAOA với trạng thái
Dicke. Sau lựa chọn, bộ tối ưu cổ điển xác định tỷ trọng theo điều kiện long-only, tổng
tỷ trọng bằng một, tối đa 40% mỗi mã, giới hạn sức chứa 5% ADV và chi phí giao dịch.
Sector cap không được áp dụng vì security master chưa có metadata ngành đáng tin cậy.

Walk-forward backtest hoàn thành 12/12 folds. Giai đoạn ngoài mẫu thực tế kéo dài từ
02/05/2022 đến 30/12/2025. Hệ thống đồng thời chạy ablation study, 300 trường hợp
sensitivity và block bootstrap có hiệu chỉnh Holm.

## 3. Kết quả tín hiệu và solver

Rank IC trung bình của XGBoost đạt 0,0795; trung vị 0,0965; dao động từ -0,2924 đến
0,3870. Tín hiệu có khả năng xếp hạng dương trung bình nhưng thiếu ổn định giữa các fold.

XY-QAOA duy trì feasibility rate 100%, trong khi penalty-QAOA đạt khoảng 64,66%.
Optimality gap trung bình của XY-QAOA là 8,93%; simulated annealing tìm được nghiệm
tham chiếu trên các instance nhỏ. Chỉ chênh lệch feasibility giữa XY-QAOA và
penalty-QAOA có ý nghĩa thống kê sau hiệu chỉnh nhiều kiểm định; chưa có bằng chứng về
quantum advantage hoặc ưu thế tốc độ lượng tử.

## 4. Hiệu quả danh mục

Pipeline đầy đủ XY-QAOA đạt lợi nhuận tích lũy -20,69%, lợi nhuận thường niên hóa
-20,06%, volatility 30,76%, Sharpe -0,6693, Sortino -0,8551 và maximum drawdown
-35,00%. Vì vậy, hệ thống hoàn thành đúng về mặt kỹ thuật nhưng cấu hình đầy đủ không
tạo hiệu quả tài chính dương trên Data A.

Trong các cấu hình đối chứng, liquidity top-K exact có kết quả tốt nhất với lợi nhuận
tích lũy 10,32% và Sharpe 0,3648. Minimum variance giảm mức sụt giảm tối đa xuống
-12,26% nhưng lợi nhuận tích lũy vẫn ở mức -1,49%. Các so sánh lợi nhuận chính không
đạt ý nghĩa thống kê sau hiệu chỉnh Holm.

## 5. Rổ cổ phiếu fold cuối

Danh mục cuối gồm SMB 40,00%, DXG 40,00%, GAS 11,39% và STB 8,61%. Danh mục tuân thủ
long-only, full investment, giới hạn 40% mỗi mã và giới hạn thanh khoản trên quy mô vốn
100 triệu đồng.

## 6. Kết luận

Data A đủ điều kiện để tái lập toàn bộ pipeline khám phá và kiểm thử các giả thuyết về
chất lượng tín hiệu, khả năng duy trì cardinality và chất lượng nghiệm. Kết quả không hỗ
trợ kết luận rằng pipeline AI–Quantum vượt trội về hiệu quả đầu tư. Data A không được
trình bày như kiểm định confirmatory toàn HOSE cho đến khi hoàn thiện điều chỉnh corporate
actions, benchmark total-return, dữ liệu ngành và các hợp đồng point-in-time còn thiếu.

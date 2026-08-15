# Báo cáo hệ thống Data A

## 1. Nhận dạng phiên bản

| Thuộc tính | Giá trị |
| --- | --- |
| Phiên bản hệ thống | Data A – complete-case exploratory pipeline |
| Thời điểm chạy xong | **16:57, ngày 13/08/2026 (Asia/Bangkok)** |
| Experiment ID | `20260813T164535-21c9b569ce` |
| Commit mã nguồn khi chạy | `19d9677ab49ebb3065d759ff12d0fe9555bb6dc2` |
| Commit công bố mốc Data A | `c01b8eb` |
| Hash panel giá | `6e046b509fef366681866328d5bd99ec63541c2de8597f0e7bebc101813baa05` |
| Trạng thái | Hoàn thành 12/12 fold; kết quả khám phá |

## 2. Mục tiêu của mốc Data A

Data A là phiên bản hệ thống đầu tiên chạy trọn vẹn pipeline AI–Quantum
trên panel giá cổ phiếu thật thay cho fixture. Mục tiêu của mốc này là kiểm tra
khả năng vận hành end-to-end, truy vết dữ liệu, duy trì ràng buộc cardinality,
so sánh các solver và tạo đầy đủ artifact nghiên cứu. Data A không được
thiết kế để bảo đảm lợi nhuận dương.

## 3. Dữ liệu và thiết kế walk-forward

Panel complete-case gồm 565.471 quan sát của 394 mã trong giai đoạn
02/01/2020–31/12/2025. Data-quality đạt `PASS` đối với schema, quan hệ OHLC,
giá trị âm, trùng khóa và timestamp khả dụng. Hệ thống loại 51 mã không đạt
độ phủ toàn kỳ, giữ tối đa 300 mã trong universe động tại mỗi ngày
quyết định.

Backtest hoàn thành 12 fold được lấy mẫu trong khoảng ngoài mẫu
02/05/2022–30/12/2025. Mỗi fold tách training, validation và test theo thứ tự thời
gian; các bước imputation, scaling, huấn luyện và ước lượng covariance chỉ
sử dụng thông tin trước ngày ra quyết định.

## 4. Kiến trúc và thuật toán

1. Các đặc trưng momentum, xu hướng, volatility, downside risk và thanh khoản
   được tính theo nguyên tắc point-in-time.
2. XGBoost tạo tín hiệu xếp hạng cổ phiếu. EWMA được dùng làm tín hiệu
   đối chứng và ước lượng ma trận hiệp phương sai đa biến.
3. Adaptive universe reduction kết hợp tín hiệu, thanh khoản, rủi ro và
   tương quan để giảm universe xuống 8 ứng viên.
4. QUBO yêu cầu chọn đúng 4 tài sản. Cùng instance được giải bằng exact
   solver, simulated annealing, penalty-QAOA và feasible-subspace XY-QAOA khởi tạo
   Dicke.
5. Tỷ trọng được tối ưu bằng phương pháp cổ điển với long-only, tổng
   tỷ trọng bằng 1, trần 40% mỗi mã, giới hạn 5% ADV và chi phí giao dịch.
6. Hệ thống chạy ablation, 300 trường hợp sensitivity và block bootstrap có
   hiệu chỉnh Holm.

## 5. Kết quả chính

| Chỉ tiêu | Data A full pipeline |
| --- | ---: |
| Lợi nhuận lũy kế sau chi phí | **-20,69%** |
| Lợi nhuận thường niên hóa | -20,06% |
| Volatility thường niên hóa | 30,76% |
| Sharpe ratio | -0,6693 |
| Sortino ratio | -0,8551 |
| Maximum drawdown | -35,00% |
| Turnover một chiều | 22,21 |
| Tổng chi phí ghi nhận | 4,81% |

XGBoost đạt Rank IC trung bình 0,0795, nhưng tín hiệu thiếu ổn định giữa
các fold. XY-QAOA đạt feasibility rate 100%, so với 64,66% của penalty-QAOA.
Optimality gap trung bình của nghiệm XY-QAOA chính là 8,93%. Chỉ chênh lệch
feasibility giữa XY-QAOA và penalty-QAOA có ý nghĩa thống kê; H3 là giả
thuyết duy nhất được hỗ trợ trên các kiểm định đã khai báo.

Liquidity top-K exact là đối chứng tốt nhất, đạt lợi nhuận 10,32% và
Sharpe 0,3648. Minimum Variance đạt -1,49%, trong khi Equal Weight toàn universe
đạt -11,80%. Các so sánh hiệu quả tài chính không có ý nghĩa thống kê
sau hiệu chỉnh Holm.

## 6. Rổ cổ phiếu fold cuối

Tại ngày quyết định 28/11/2025, Data A lựa chọn:

| Mã | Tỷ trọng |
| --- | ---: |
| SMB | 40,00% |
| DXG | 40,00% |
| GAS | 11,39% |
| STB | 8,61% |

Danh mục tuân thủ cardinality bằng 4, long-only, full investment, trần 40% mỗi
mã và giới hạn thanh khoản ở quy mô vốn giả định 100 triệu đồng.

## 7. Hạn chế và lý do hình thành Data B

Data A chứng minh pipeline có thể vận hành trên dữ liệu thật, nhưng còn bộc lộ
các hạn chế lớn: lợi nhuận âm, volatility và drawdown cao, 12 fold không
liên tục, chưa có cơ chế giữ tiền mặt theo trạng thái thị trường, chưa buộc
khả năng mua lô trước solver và chưa tận dụng validation để phối hợp nhiều
nhóm tín hiệu. Sector cap không thể áp dụng do thiếu metadata ngành.

Ngoài ra, Data A là panel complete-case nên vẫn có nguy cơ coverage/survivorship
bias; hợp đồng điều chỉnh giá, corporate actions và benchmark total-return chưa được
xác minh đầy đủ. Những vấn đề này là cơ sở để phát triển mốc Data B;
chúng không cho phép diễn giải Data A như bằng chứng quantum advantage hoặc
khuyến nghị đầu tư.

# Báo cáo hệ thống Data B

## 1. Nhận dạng phiên bản

| Thuộc tính | Giá trị |
| --- | --- |
| Phiên bản hệ thống | Data B – optimized exploratory pipeline |
| Thời điểm chạy xong | **19:48, ngày 15/08/2026 (Asia/Bangkok)** |
| Experiment ID | `20260815T184821-a800b2584d` |
| Commit mã nguồn khi chạy | `65c266d452d42b7e115a01f1fe2bfeacf69d5cf5` |
| Commit công bố mốc Data B | `ecdeac4` |
| Hash panel giá | `6e046b509fef366681866328d5bd99ec63541c2de8597f0e7bebc101813baa05` |
| Trạng thái | Hoàn thành 44/44 fold; kết quả khám phá |

## 2. Phạm vi thay đổi so với Data A

Data B là mốc tối ưu hóa lớn được xây dựng trên panel Data A bất biến.
Hash panel giá không thay đổi; vì vậy chênh lệch kết quả xuất phát từ quy trình
ra quyết định, quản trị rủi ro và ràng buộc thực thi, không phải từ việc thay
dữ liệu sau khi quan sát kết quả.

Những thay đổi kiến trúc chính gồm:

1. Tăng từ 12 fold lấy mẫu lên 44 fold theo tháng liên tục.
2. Sử dụng 24 tháng train, 3 tháng validation, 1 tháng test và embargo 20 ngày.
3. Trộn XGBoost với technical factor; tỷ lệ trộn được chọn riêng trên
   validation của mỗi fold.
4. Hiệu chỉnh tín hiệu sang thang lợi nhuận 20 ngày trước khi lập QUBO.
5. Bổ sung ngưỡng lợi nhuận kỳ vọng, bộ lọc thanh khoản/rủi ro, cụm tương
   quan và tính ổn định giữa các kỳ cho adaptive universe reduction.
6. Loại mã không thể mua ít nhất một lô 100 trước khi đưa vào solver.
7. Dùng nghiệm khả thi có energy tốt nhất đã quan sát qua các shot XY-QAOA
   cho danh mục triển khai.
8. Bổ sung market-regime exposure và tiền mặt thay vì luôn full investment.
9. Đưa lô 100, quy mô vốn, ADV và residual cash vào audit thực thi.

## 3. Dữ liệu và thiết kế thực nghiệm

Data B tái sử dụng 565.471 quan sát của 394 mã trong giai đoạn
02/01/2020–31/12/2025. Giai đoạn ngoài mẫu thực tế là
02/05/2022–30/12/2025. Toàn bộ 44 fold được hoàn thành; các bước fit imputer,
scaler, XGBoost, lựa chọn tỷ lệ blend, hiệu chỉnh tín hiệu và ước lượng
covariance không sử dụng test window.

EWMA covariance là mô hình đa biến, dùng span 60 và chân trời 20 ngày. AUR
giảm universe xuống 8 ứng viên trong 43 fold và 7 ứng viên trong một fold.
QUBO chọn đúng 4 mã. Solver gồm exact, simulated annealing, penalty-QAOA và
feasible-subspace XY-QAOA khởi tạo Dicke trên ideal statevector simulator.

## 4. Kết quả chính

| Chỉ tiêu | Data B full pipeline |
| --- | ---: |
| Lợi nhuận gộp lũy kế | 20,75% |
| Lợi nhuận lũy kế sau chi phí | **16,13%** |
| Lợi nhuận thường niên hóa | 4,08% |
| Volatility thường niên hóa | 7,54% |
| Sharpe ratio | 0,1761 |
| Sortino ratio | 0,1753 |
| Maximum drawdown | -14,05% |
| Fold dương/âm | 25/19 |
| Turnover một chiều | 18,62 |
| Tổng chi phí ghi nhận | 3,90% |

Minimum Variance đạt 28,16%, liquidity top-K exact đạt 19,35% và Equal Weight
toàn universe đạt 6,95%. Data B vượt Equal Weight và Markowitz Mean–Variance
về lợi nhuận mô tả, nhưng không vượt Minimum Variance. Các chênh lệch
lợi nhuận chính không có ý nghĩa thống kê sau hiệu chỉnh Holm.

Rank IC trung bình của XGBoost là 0,0599; Rank IC của tín hiệu trộn là 0,0534
và EWMA signal đối chứng là -0,0027. Chênh lệch XGBoost–EWMA chưa có ý
nghĩa thống kê. AUR cũng không cải thiện có ý nghĩa về lợi nhuận kỳ tiếp
theo hoặc đa dạng hóa so với fixed top-M.

XY-QAOA đạt feasibility rate 100%, trong khi penalty-QAOA đạt 56,95%. Optimality
gap trung bình của bitstring XY-QAOA có xác suất cao nhất là 3,40%. Nghiệm khả
thi tốt nhất đã quan sát và được dùng để triển khai trùng giá trị exact trong
44/44 fold. Chỉ H3 về feasibility được hỗ trợ thống kê; kết quả không
chứng minh quantum advantage.

## 5. Rổ cổ phiếu fold cuối

Tại ngày quyết định 28/11/2025, hệ thống xác định trạng thái `risk_off`.
Mức giải ngân mục tiêu là 25%; sau khi làm tròn lô 100 trên danh mục 100
triệu đồng, kết quả thực thi là:

| Mã | Tỷ trọng thực thi | Số cổ phiếu |
| --- | ---: | ---: |
| SMB | 8,0400% | 200 |
| KOS | 3,8950% | 100 |
| HTG | 3,8917% | 100 |
| TLD | 3,2760% | 400 |

Tổng tỷ trọng cổ phiếu thực thi là 19,10%, tiền mặt thực tế là 80,90%.
Lợi nhuận 20 ngày kỳ vọng của danh mục sau làm tròn lô là **+0,1237%**.
Đây là kỳ vọng ex-ante; lợi nhuận thực tế của fold tháng 12/2025 là
**-0,2595%** sau chi phí.

## 6. Nguyên nhân lợi nhuận chuyển sang dương

Data B dương chủ yếu do tích hợp validation-selected signal blend, tăng vai trò
thanh khoản, dùng nghiệm XY-QAOA tốt nhất đã quan sát, buộc khả năng mua lô và
áp dụng market-regime exposure. Yếu tố có ảnh hưởng lớn nhất là tiền mặt:
mức giải ngân mục tiêu trung bình chỉ 30,57%, còn tỷ trọng thực thi trung
bình sau lô là 26,64%. Nhờ đó volatility và drawdown giảm mạnh so với Data A.

Tuy nhiên, lợi nhuận chưa bền vững. Hai fold tháng 7 và 8/2023 đạt lần lượt
6,68% và 14,98%. Nếu loại hai fold này, phần còn lại có lợi nhuận cộng dồn
khoảng -5,33%. Vì vậy lợi nhuận dương không được diễn giải là alpha ổn
định hoặc hiệu quả do riêng thành phần lượng tử.

## 7. So sánh mốc Data A và Data B

| Thuộc tính | Data A | Data B |
| --- | ---: | ---: |
| Số fold | 12 | 44 |
| Lợi nhuận lũy kế | -20,69% | **16,13%** |
| Lợi nhuận thường niên | -20,06% | 4,08% |
| Volatility thường niên | 30,76% | 7,54% |
| Sharpe | -0,6693 | 0,1761 |
| Maximum drawdown | -35,00% | -14,05% |
| Tỷ trọng cổ phiếu | Full investment | Trung bình thực thi 26,64% |
| Giả thuyết được hỗ trợ | H3 | H3 |

So sánh này có tính mô tả, không phải phép nhận dạng nhân quả hoàn chỉnh,
vì Data A dùng 12 fold lấy mẫu trong khi Data B dùng 44 fold liên tục và có cơ
chế exposure khác nhau.

## 8. Hạn chế bắt buộc

1. Panel complete-case vẫn có nguy cơ coverage/survivorship bias.
2. Hợp đồng điều chỉnh giá và corporate actions chưa được xác minh đầy đủ.
3. Chưa có benchmark total-return đã xác minh.
4. Sector cap chưa áp dụng do metadata ngành không đủ.
5. XY-QAOA chạy trên ideal statevector simulator, không phải thiết bị NISQ thực.
6. Data B được phát triển sau khi đã quan sát Data A; cần một holdout mới
   hoặc dữ liệu tương lai để kiểm định confirmatory.
7. Lợi nhuận dương chưa có ý nghĩa thống kê và phụ thuộc đáng kể vào
   hai fold tốt trong năm 2023.

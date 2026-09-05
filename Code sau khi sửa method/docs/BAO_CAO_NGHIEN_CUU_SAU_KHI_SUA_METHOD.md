# SO SÁNH ADAPTIVE VÀ QUANTUM-ASSISTED UNIVERSE REDUCTION TRONG KHUNG TỐI ƯU DANH MỤC LAI

## Tóm tắt

Nghiên cứu xây dựng một thiết kế có kiểm soát để so sánh Adaptive Universe
Reduction (AUR) và Quantum-Assisted Universe Reduction (QAUR) trong bài toán
tối ưu danh mục cổ phiếu. Hai phương pháp bắt đầu từ cùng point-in-time universe,
cùng tín hiệu lợi suất XGBoost và cùng ước lượng rủi ro EWMA, sau đó tạo hai tập
ứng viên Top-K có cùng kích thước. Mỗi tập được đưa qua chính xác cùng một pipeline
downstream gồm cardinality-constrained QUBO, feasible-subspace XY-QAOA, tối ưu
tỷ trọng cổ điển và walk-forward backtest. Thiết kế này cô lập tác động của bước
giảm vũ trụ thay vì nhầm lẫn AUR với một quantum portfolio solver.

Thực nghiệm tái sử dụng bộ forecast point-in-time của 12 fold hiện có trên dữ liệu
HOSE 2020-2025. Với Top-8 và cardinality cuối bằng bốn, hai reducer có Jaccard
similarity trung bình 0,754. Trên 255 phiên ngoài mẫu, AUR đạt lợi nhuận tích lũy
1,477%, còn QAUR đạt 1,463%. Chênh lệch lợi suất ngày không có ý nghĩa thống kê
(paired t-test p=0,995; Wilcoxon p=0,865). Cả hai nhánh có feasibility rate 100%
ở solver downstream. Kết quả xác nhận framework so sánh vận hành đúng nhưng chưa
cho thấy QAUR vượt AUR. Backend QAUR hiện là classical cardinality-preserving
surrogate cho quantum-ready QUBO; vì vậy nghiên cứu không tuyên bố quantum advantage.

**Từ khóa:** universe reduction; asset pre-selection; XGBoost; EWMA; QUBO;
XY-QAOA; cardinality constraint; walk-forward backtest; HOSE.

## 1. Giới thiệu

Portfolio optimization quy mô lớn gặp hai lớp khó khăn khác nhau. Lớp thứ nhất
là hình thành một candidate universe đủ chất lượng từ hàng trăm tài sản có chất
lượng tín hiệu, thanh khoản và cấu trúc tương quan khác nhau. Lớp thứ hai là chọn
một tập tài sản thỏa cardinality và phân bổ tỷ trọng dưới các ràng buộc giao dịch.
Nếu hai lớp này bị trộn, kết quả tốt hoặc xấu không thể được quy chính xác cho
reducer, solver hay weight optimizer.

Giới hạn qubit làm vấn đề nhận diện trở nên quan trọng hơn. Phần lớn quantum
portfolio experiments chỉ xử lý instance nhỏ hoặc giả định candidate set đã có
sẵn. Một reducer cổ điển có thể chọn ứng viên tuần tự theo score, trong khi một
quantum-assisted reducer có thể mã hóa đồng thời chất lượng đơn biến và redundancy
cặp dưới cardinality constraint. Tuy nhiên, so sánh hai reducer chỉ hợp lệ khi
mọi đầu vào và mọi bước downstream được giữ cố định.

Nghiên cứu vì vậy không hỏi “Adaptive hay QAOA tốt hơn?”. AUR là một universe
reducer; XY-QAOA là một constrained portfolio solver. Câu hỏi hợp lệ là AUR hay
QAUR tạo candidate set tốt hơn, và khác biệt đó có tồn tại sau khi cả hai tập đi
qua cùng một XY-QAOA và cùng weight optimizer hay không.

## 2. Research gap

Các nghiên cứu sparse portfolio và asset pre-selection cho thấy có thể giảm chi
phí và độ phức tạp bằng cách giới hạn số tài sản. Các nghiên cứu decomposition
gần đây tiếp tục cho thấy correlation-based partitioning có thể làm các subproblem
phù hợp với thiết bị lượng tử gần hạn. Song song, Quantum Alternating Operator
Ansatz và XY mixers cung cấp cách duy trì hard cardinality trong feasible subspace.
Các nghiên cứu hybrid và backtesting lượng tử đã đánh giá solver trên dữ liệu thật.

Tuy nhiên, bốn dòng nghiên cứu trên thường không tạo ra một controlled experiment
để nhận diện riêng tác động của universe reduction. Candidate set thường được giả
định trước, được tạo bởi một heuristic duy nhất, hoặc thay đổi đồng thời với solver.
Do đó, khác biệt out-of-sample có thể bị confound bởi forecast, reducer, solver,
allocation và transaction cost.

Khoảng trống nghiên cứu cụ thể là thiếu một pipeline mà AUR và QAUR nhận cùng
point-in-time universe, XGBoost/EWMA outputs và cùng K, được so sánh trực tiếp ở
tầng reduction, rồi được đánh giá qua một downstream pipeline hoàn toàn giống
nhau. Nghiên cứu hiện tại lấp khoảng trống thiết kế này; nó không mặc định chứng
minh lợi thế lượng tử.

## 3. Câu hỏi, mục tiêu và giả thuyết

### 3.1. Câu hỏi nghiên cứu

- RQ1: AUR và QAUR tạo ra các Top-K khác nhau như thế nào về chất lượng, redundancy,
  stability và turnover?
- RQ2: Khi downstream được giữ cố định, QAUR có cải thiện hiệu quả danh mục ngoài
  mẫu so với AUR không?
- RQ3: Lợi ích tiềm năng có đủ bù runtime, resource demand và độ phức tạp của QAUR không?
- RQ4: Kết quả có ổn định theo K, cardinality, seed và market regime không?

### 3.2. Mục tiêu

1. Xây dựng một forecast/risk layer dùng chung và point-in-time safe.
2. Xây dựng AUR và QAUR với cùng input/output contract.
3. Đánh giá trực tiếp chất lượng hai Top-K.
4. Áp dụng cùng cardinality QUBO, XY-QAOA và weight optimizer.
5. Đánh giá out-of-sample và kiểm định chênh lệch theo cặp.
6. Báo cáo minh bạch resource limitation và không suy diễn quantum advantage.

### 3.3. Giả thuyết

- H1: QAUR tạo Top-K có objective chất lượng-redundancy tốt hơn AUR.
- H2: QAUR tạo mức đa dạng hóa cao hơn AUR tại cùng K.
- H3: QAUR không làm tăng universe turnover so với AUR.
- H4: QAUR + shared downstream pipeline tạo risk-adjusted return cao hơn AUR +
  shared downstream pipeline.
- H5: Khác biệt giữa hai nhánh bền vững qua fold, seed và cấu hình.

## 4. Phương pháp nghiên cứu

### 4.1. Sáu giai đoạn

**Giai đoạn 1 - Forecasting.** XGBoost tạo expected-return signal và EWMA ước
lượng covariance. Forecast được sinh trước thời điểm test và được dùng chung.

**Giai đoạn 2 - AUR.** AUR xếp hạng thích nghi theo signal, liquidity, inverse
risk và retention stability, sau đó thêm tuần tự tài sản với correlation penalty.

**Giai đoạn 3 - QAUR.** QAUR biểu diễn universe reduction thành cardinality QUBO.
Backend hiện tại là classical surrogate giữ cardinality để kiểm chứng formulation.

**Giai đoạn 4 - Reduction comparison.** Hai Top-K được so sánh trước portfolio
optimization bằng overlap, Jaccard, objective, turnover và các đặc tính tập hợp.

**Giai đoạn 5 - Shared optimization.** Mỗi Top-K đi qua cùng portfolio QUBO,
feasible-subspace XY-QAOA và SLSQP weight allocation.

**Giai đoạn 6 - Walk-forward evaluation.** Mỗi quyết định chỉ dùng dữ liệu trước
test window; chi phí giao dịch được trừ tại ngày đầu kỳ nắm giữ.

### 4.2. Common reduction information

Với tài sản i tại thời điểm t, unary quality score dùng chung là:

q_it = a_s S_it + a_l L_it + a_r R_it + a_h H_it,

trong đó S là percentile của XGBoost signal, L là percentile thanh khoản, R là
inverse-risk percentile và H là chỉ báo tài sản được giữ từ fold trước. Các thành
phần được xây dựng từ cùng snapshot; khác biệt chỉ nằm ở search mechanism.

### 4.3. Adaptive Universe Reduction

AUR xây dựng tập A tuần tự. Ở mỗi bước, tài sản được thêm là:

i* = argmax(i không thuộc A) [q_it - gamma * sum(j thuộc A) |rho_ij,t|].

Quá trình dừng khi |A|=K. Đây là greedy adaptive screening: score thay đổi theo
forecast, liquidity, risk, lịch sử retention và cấu trúc correlation tại mỗi fold.

### 4.4. Quantum-Assisted Universe Reduction

QAUR dùng biến z_i thuộc {0,1} và giải:

max_z  sum_i q_it z_i - gamma * sum_{i<j} |rho_ij,t| z_i z_j

subject to sum_i z_i = K.

Đây là Q^UR, tách biệt với portfolio QUBO Q^PO. Lần chạy hiện tại dùng
cardinality-preserving multi-start swap search làm classical surrogate. Vì chưa
chạy QPU hoặc quantum simulator cho full-universe Q^UR, kết quả chỉ đánh giá
formulation và data flow của QAUR.

### 4.5. Shared portfolio QUBO và XY-QAOA

Với candidate set C có K phần tử, biến x_i biểu thị tài sản được chọn vào danh
mục cuối. Shared portfolio objective là:

min_x  lambda x' Sigma x - mu' x,

subject to sum_i x_i = k_p.

XY mixer bảo toàn Hamming weight nên mọi sample ở feasible subspace chứa đúng
k_p tài sản. Cùng depth, shots, seed policy và solution-selection rule được dùng
cho AUR và QAUR. Sau asset selection, tỷ trọng liên tục giải:

min_w  lambda_w w' Sigma w - mu'w,

subject to sum_i w_i=1 và l <= w_i <= u.

### 4.6. Walk-forward design

Thực nghiệm dùng 12 fold đã có, train 24 tháng, validation ba tháng và test một
tháng. Hai nhánh dùng cùng decision date, test observations, transaction cost 25
bps, Top-K=8 và final cardinality=4. Tổng số quan sát ngoài mẫu là 255 phiên.

## 5. Kết quả

### 5.1. So sánh universe reduction

Jaccard similarity trung bình giữa Top-8 AUR và QAUR là 0,754. Trong bốn fold
cuối, overlap tăng lên 8/8 ở ba fold, cho thấy hai objective hội tụ khi cấu trúc
quality-correlation tạo một tập ứng viên nổi trội. Ở các fold trước, Jaccard thấp
nhất là 0,455, đủ để downstream comparison nhận diện tác động của reducer.

Turnover trung bình của candidate-to-portfolio process là 0,817 đối với AUR và
0,809 đối với QAUR. Kết quả mô tả này phù hợp với H3 nhưng chưa đủ để kết luận
thống kê khi chỉ có 12 fold.

### 5.2. Feasibility downstream

Cả hai nhánh đạt feasibility rate 100%. Kết quả này là hệ quả của shared
feasible-subspace solver, không phải ưu thế riêng của QAUR. Việc feasibility giống
nhau là một design invariant cần thiết để cô lập universe-reduction effect.

### 5.3. Hiệu quả ngoài mẫu

| Phương pháp | Cumulative return | Annualized return | Annualized volatility | Sharpe (rf=0) | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| AUR + shared pipeline | 1,477% | 1,460% | 27,839% | 0,052 | -25,889% |
| QAUR + shared pipeline | 1,463% | 1,445% | 27,785% | 0,052 | -25,889% |

Full-Universe equal-weight được xuất như một feasibility baseline riêng. Vì nhánh
này không đi qua XY-QAOA 8 qubit, nó không thuộc controlled AUR-vs-QAUR test và
không được dùng để kết luận về lợi ích lượng tử.

Chênh lệch lợi suất ngày QAUR trừ AUR là -0,00000114. Paired t-test cho p=0,995;
Wilcoxon signed-rank test cho p=0,860. Không có bằng chứng bác bỏ giả thuyết không
về hiệu quả bằng nhau. H4 không được ủng hộ trong lần chạy hiện tại.

## 6. Thảo luận

Kết quả quan trọng nhất là tính nhận diện của thiết kế mới. Feasibility và weight
allocation được cố định, vì vậy khác biệt nhỏ trong return có thể được truy về
candidate universe thay vì thay đổi solver. Logic cũ “Adaptive vs QAOA” không cho
phép diễn giải này vì so sánh hai thành phần ở hai tầng khác nhau.

QAUR không vượt AUR trong dữ liệu hiện tại. Hai nguyên nhân có thể đồng thời tồn
tại. Thứ nhất, unary score của hai reducer giống nhau và correlation penalty tương
đối nhỏ, nên greedy search đã gần nghiệm tốt của Q^UR. Thứ hai, downstream chọn
bốn trong tám tài sản, có thể hấp thụ một phần khác biệt ở candidate level. Vì
vậy nghiên cứu tiếp theo cần sensitivity theo K, k_p và gamma thay vì chỉ tăng
độ phức tạp quantum backend.

Không được diễn giải feasibility 100% là quantum advantage. Nó đến từ feasible
subspace theo thiết kế. QAUR full-universe chưa chạy trên QPU; XY-QAOA downstream
hiện là statevector surrogate. Exact/classical benchmarks và noise-aware runs là
điều kiện cần trước các tuyên bố mạnh hơn.

## 7. Hạn chế

- Forecast được tái sử dụng từ experiment hiện có thay vì huấn luyện lại trong
  thư mục mới; điều này bảo đảm common input nhưng tạo dependency vào artifact cũ.
- QAUR dùng classical surrogate cho quantum-ready QUBO.
- XY-QAOA implementation hiện là feasible-subspace statevector surrogate, chưa
  phải circuit execution trên quantum hardware.
- Chỉ có 12 fold và 255 phiên ngoài mẫu; power thống kê còn hạn chế.
- Full-Universe equal-weight đã được chạy nhưng chỉ là equivalent classical
  baseline; chưa có Full-Universe XY-QAOA do giới hạn state-space/qubit.
- Historical sector classification chưa đủ point-in-time để đánh giá sector coverage.

## 8. Kết luận

Nghiên cứu đã tái định nghĩa đúng đối tượng so sánh là AUR và QAUR, trong khi
XY-QAOA trở thành shared downstream solver. Trên 12 fold, hai reducer tạo candidate
set khác nhau vừa phải nhưng cho kết quả đầu tư gần như giống nhau. Không có bằng
chứng QAUR vượt AUR và không có cơ sở tuyên bố quantum advantage. Đóng góp hiện
tại là một framework có kiểm soát, tái lập được và có thể mở rộng sang quantum
backend thật mà không thay đổi các thành phần downstream.

## Tài liệu tham khảo

Danh mục và phân tích nguồn chi tiết nằm trong `RESEARCH_GAP_VA_TAI_LIEU.md`.
Các nguồn cốt lõi gồm Markowitz (1952), Hadfield et al. (2019), Wang et al.
(2020), Slate et al. (2021), Mugel et al. (2021), Brandhofer et al. (2023), He
et al. (2023), Carrascal et al. (2023) và Acharya et al. (2024).

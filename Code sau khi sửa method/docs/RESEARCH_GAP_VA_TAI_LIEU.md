# Cơ sở tài liệu và research gap cho framework AUR vs QAUR

## Lưu ý về thuật ngữ

“Adaptive Universe Reduction” chưa phải tên của một dòng phương pháp đã được
chuẩn hóa rộng rãi. Trong bài này, AUR là tên vận hành cho cơ chế screening động
kết hợp tín hiệu dự báo, thanh khoản, rủi ro, độ ổn định và tương quan. Nền tảng
học thuật gần nhất của AUR là asset pre-selection, sparse/cardinality-constrained
portfolio selection, clustering và decomposition. Vì vậy không được viết rằng
literature đã thiết lập một thuật toán chuẩn có tên AUR.

## Các dòng nghiên cứu trực tiếp liên quan

1. **Portfolio thưa và asset pre-selection.** Maringer và Oyewumi (2007) cho
   thấy index tracking có thể dùng một tập con tài sản dưới cardinality constraint.
   Dòng nghiên cứu này chứng minh giá trị kinh tế của việc giảm số tài sản nhưng
   chủ yếu giải selection và allocation trong cùng một bài toán, chưa tách một
   universe-reduction layer thích nghi để cấp đầu vào công bằng cho nhiều solver.

2. **Decomposition phục vụ thiết bị lượng tử gần hạn.** Acharya et al. (2024)
   kết hợp random-matrix preprocessing, spectral clustering và risk rebalancing,
   giảm khoảng 80% kích thước subproblem. Công trình này là bằng chứng trực tiếp
   rằng giới hạn thiết bị đòi hỏi một tầng giảm/decomposition có chủ đích. Tuy
   nhiên, decomposition tạo nhiều subproblem rồi tổng hợp, khác với câu hỏi của
   nghiên cứu hiện tại: cùng một universe và forecast, phương pháp giảm cổ điển
   hay quantum-assisted nào tạo Top-K tốt hơn trước một optimizer downstream cố định?

3. **Hybrid quantum-classical portfolio optimization.** Mugel et al. (2021,
   2022) xử lý danh mục động, holding-period và dữ liệu thật bằng kiến trúc lai;
   Slate et al. (2021) nghiên cứu quantum-walk portfolio optimization. Các nghiên
   cứu này chứng minh tính phù hợp của phân tách classical/quantum, nhưng trọng tâm
   vẫn là solver/portfolio objective, không phải controlled comparison giữa hai
   universe reducers dùng chung forecast và downstream optimizer.

4. **Constrained QAOA và feasible subspace.** Hadfield et al. (2019) mở rộng
   QAOA thành Quantum Alternating Operator Ansatz; Wang et al. (2020) phân tích
   XY mixers để giữ hard constraints; He et al. (2023) cho thấy alignment giữa
   initial state và mixer ảnh hưởng chất lượng QAOA, kể cả trên portfolio benchmark.
   Các công trình này biện minh cho việc dùng XY-QAOA downstream, nhưng không biến
   XY-QAOA thành đối thủ của AUR: chúng nằm ở hai tầng quyết định khác nhau.

5. **Benchmarking và backtesting.** Brandhofer et al. (2023) nhấn mạnh việc
   benchmark QAOA portfolio optimization cẩn thận; Carrascal et al. (2023) thực
   hiện so sánh backtest classical/quantum. Tuy nhiên, nếu mỗi solver nhận một
   candidate set khác nhau thì không thể quy chênh lệch out-of-sample cho tầng
   reduction. Đây là confounding mà thiết kế mới cần loại bỏ.

## Research gap được xác định

Literature hiện có đã nghiên cứu riêng lẻ (i) tiền chọn hoặc tạo danh mục thưa,
(ii) decomposition để phù hợp ngân sách qubit, (iii) QUBO/QAOA cho asset selection,
và (iv) hybrid portfolio backtesting. Khoảng trống không nằm ở việc thiếu thêm
một lần so sánh “Adaptive với QAOA”. Khoảng trống nằm ở việc thiếu một thiết kế
thực nghiệm end-to-end có kiểm soát, trong đó một reducer thích nghi cổ điển và
một reducer quantum-assisted:

- bắt đầu từ cùng point-in-time universe;
- nhận cùng forecast lợi suất XGBoost và risk estimate EWMA;
- tạo hai candidate set có cùng kích thước Top-K;
- được đánh giá trực tiếp về retained signal, redundancy, stability, turnover,
  runtime và resource demand;
- sau đó đi qua cùng cardinality QUBO, cùng XY-QAOA, cùng weight optimizer và
  cùng walk-forward protocol.

Thiết kế này tách được **universe-reduction effect** khỏi **solver effect** và
**weight-allocation effect**. Đây là đóng góp phương pháp chính. QAUR chỉ nên được
gọi là quantum-assisted khi formulation/backend thật sự có thành phần lượng tử;
trong lần chạy hiện tại, QUBO của QAUR dùng classical surrogate nên kết quả chỉ
là kiểm chứng framework, không phải quantum advantage.

## Tài liệu cốt lõi đã kiểm chứng

- Acharya, A., et al. (2024). *Decomposition Pipeline for Large-Scale Portfolio
  Optimization with Applications to Near-Term Quantum Computing*.
  https://arxiv.org/abs/2409.10301
- Brandhofer, S., et al. (2023). *Benchmarking the performance of portfolio
  optimization with QAOA*. Quantum Information Processing, 22, 25.
  https://doi.org/10.1007/s11128-022-03766-5
- Carrascal, G., et al. (2023). *Backtesting Quantum Computing Algorithms for
  Portfolio Optimization*. IEEE Transactions on Quantum Engineering.
  https://tqe.ieee.org/2023/11/28/backtesting-quantum-computing-algorithms-for-portfolio-optimization/
- Hadfield, S., et al. (2019). *From the Quantum Approximate Optimization
  Algorithm to a Quantum Alternating Operator Ansatz*. Algorithms, 12(2), 34.
  https://doi.org/10.3390/a12020034
- He, Z., et al. (2023). *Alignment between initial state and mixer improves
  QAOA performance for constrained optimization*. npj Quantum Information, 9, 121.
  https://doi.org/10.1038/s41534-023-00787-5
- Maringer, D., & Oyewumi, O. (2007). *Index tracking with constrained
  portfolios*. Intelligent Systems in Accounting, Finance and Management, 15, 57-71.
  https://doi.org/10.1002/isaf.285
- Mugel, S., et al. (2021). *Hybrid quantum investment optimization with minimal
  holding period*. Scientific Reports, 11, 19587.
  https://doi.org/10.1038/s41598-021-98297-x
- Slate, N., et al. (2021). *Quantum walk-based portfolio optimisation*.
  Quantum, 5, 513. https://doi.org/10.22331/q-2021-07-28-513
- Wang, Z., et al. (2020). *XY mixers: Analytical and numerical results for the
  quantum alternating operator ansatz*. Physical Review A, 101, 012320.
  https://doi.org/10.1103/PhysRevA.101.012320


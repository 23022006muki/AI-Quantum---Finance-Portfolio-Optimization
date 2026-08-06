# Quantum implementation

## Bài toán và biểu diễn

Tầng lượng tử nhận tập ứng viên đã thu hẹp và xây dựng QUBO từ vector lợi nhuận kỳ vọng cùng ma trận hiệp phương sai EWMA hoặc Ledoit–Wolf. Hệ thống lưu cả ma trận QUBO và biểu diễn Ising gồm offset, trường tuyến tính và tương tác cặp. Test tự động xác nhận năng lượng của mọi bitstring nhỏ không đổi sau ánh xạ.

## XY-QAOA khả thi

XY-QAOA chạy trên ideal statevector simulator trong không gian fixed-Hamming-weight. Trạng thái đầu là phân bố đều trên các bitstring có đúng K bit một, tương đương Dicke state trong không gian con. XY exchange mixer chỉ nối các trạng thái khác nhau bởi một phép đổi 1–0, vì vậy cardinality được bảo toàn theo cấu trúc trong mô phỏng lý tưởng.

## Tối ưu tham số

Góc gamma và beta được tối ưu bằng COBYLA đa khởi tạo. Chi phí pha được scale để tránh pha gần như bằng không khi QUBO có hệ số nhỏ; objective báo cáo vẫn là năng lượng kinh tế gốc. Mỗi run lưu seed, budget, tham số đầu/cuối, objective trace, số evaluation và stopping reason.

Penalty-QAOA sử dụng cùng họ optimizer, depth, shots và budget khi so sánh chính. Penalty strength được lưu trong artifact. Exact và simulated annealing là comparator cổ điển, không được mô tả là thuật toán lượng tử.

## Quy tắc chọn nghiệm

Nghiệm chính của XY-QAOA là bitstring có xác suất cao nhất trong phân bố cuối. Đối với penalty-QAOA, nghiệm chính là bitstring khả thi có xác suất cao nhất. Nghiệm khả thi có năng lượng thấp nhất từng xuất hiện trong shots được lưu riêng dưới tên `best_observed`; nó không thay thế nghiệm chính khi tính primary optimality gap.

Artifact solver còn lưu expected energy, primary probability, feasibility rate, probability of optimum, bitstring counts và optimizer trace. Ngoài ideal statevector, sensitivity có thể áp dụng kênh depolarizing và readout ở mức mô phỏng hiện tượng học. `uniform_probability_noise_proxy` là phép trộn phân bố cũ được gắn nhãn riêng. Không cấu hình nào trong số này là gate-level hardware noise đã hiệu chuẩn hoặc bằng chứng từ thiết bị NISQ thực.


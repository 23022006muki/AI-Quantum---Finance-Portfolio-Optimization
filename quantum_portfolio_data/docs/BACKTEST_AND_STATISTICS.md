# Backtest and statistical protocol

## Walk-forward

Mỗi fold gồm training, validation và test theo thứ tự thời gian. Nhãn là lợi nhuận forward; `label_end_time` phải nằm hoàn toàn trước biên tiếp theo sau embargo. Imputer, scaler, feature coverage và XGBoost chỉ được fit bằng training data. Validation được dùng để chọn hyperparameter; test không tham gia tuning.

## Accounting

Mỗi chiến lược giao dịch tại cùng lịch tái cân bằng. Turnover được tính trên hợp của vị thế cũ và mới. Chi phí tỷ lệ được khấu trừ tại thời điểm tái cân bằng. Giữa hai lần tái cân bằng, hệ thống giữ số đơn vị tương đối của từng tài sản và để tỷ trọng drift theo lợi nhuận; không ngầm tái cân bằng hằng ngày. Gross return, net return, trades, pre-trade weights, target weights và cost ledger được lưu riêng.

## Benchmark và ablation

Các benchmark độc lập gồm equal-weight toàn universe, equal-weight tập ứng viên, Markowitz mean–variance, minimum variance, liquidity Top-K, EWMA Top-K, XGBoost Top-K, adaptive+exact, adaptive+SA, adaptive+penalty-QAOA và adaptive+XY-QAOA. Tất cả chịu cùng policy chi phí. Ablation thay đổi selector hoặc solver trong khi giữ lịch và accounting cố định.

## Thống kê

So sánh lợi nhuận dùng paired moving-block bootstrap để bảo toàn phụ thuộc chuỗi thời gian. Phân bố null được tạo sau khi center chênh lệch về không; báo cáo effect size, khoảng tin cậy và p-value hai phía. Holm correction được áp dụng cho họ so sánh. H1 so sánh Rank IC theo fold giữa XGBoost và EWMA; H5 so sánh net return của pipeline với benchmark.

Sensitivity phải thực sự chạy lại solver hoặc accounting cho các mức depth, shots, cardinality, noise proxy và transaction cost; không chỉ thay nhãn cột. Kết luận chỉ có hiệu lực trong grid và seeds đã khai báo.


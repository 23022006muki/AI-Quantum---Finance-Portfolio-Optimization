# Research limitations and claim boundaries

1. Không có quantum advantage claim. Simulator cổ điển không chứng minh ưu thế tốc độ, khả năng mở rộng hay chống nhiễu của phần cứng lượng tử.
2. Research run bị chặn cho đến khi có historical HOSE universe, corporate actions và membership/event provenance hợp lệ. Chuỗi giá thật đơn thuần không đủ để kiểm soát survivorship bias.
3. Nguồn FinanceDataReader/Yahoo/vnstock có thể hữu ích cho giá, nhưng ngày đầu có giá không phải bằng chứng về ngày niêm yết và không được dùng thay listing history.
4. Point-in-time fundamentals, macro và foreign flow bị loại theo fold khi coverage hoặc provenance không đủ. Điều này giới hạn phạm vi diễn giải vai trò của các biến ngoài thị trường.
5. Exact solver chỉ khả thi ở universe lượng tử nhỏ và được dùng làm oracle đánh giá, không phải bằng chứng về scalability.
6. Kết quả phụ thuộc giai đoạn, chi phí, lịch tái cân bằng, target horizon, feature set, optimizer budget và random seeds. Sensitivity không bao quát mọi cấu hình.
7. Backtest không mô hình hóa đầy đủ market impact, giới hạn biên độ, khớp lệnh từng phần, thuế, phí tối thiểu, room ngoại hay đình chỉ giao dịch nếu dữ liệu tương ứng chưa được cung cấp.
8. Demo fixture chỉ kiểm chứng luồng phần mềm; không được trích làm kết quả thực nghiệm HOSE.

Mọi báo cáo phải công bố requested/actual data range, actual out-of-sample range, số fold hoàn thành, data class, provenance hash, leakage status và giới hạn nêu trên.

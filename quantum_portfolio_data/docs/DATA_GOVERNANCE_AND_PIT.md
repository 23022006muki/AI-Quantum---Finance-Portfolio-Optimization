# Data governance and point-in-time policy

## Nguyên tắc quyết định

Mỗi quan sát phải phân biệt thời điểm kinh tế được quan sát (`date`, `observation_date`, `fiscal_period_end`, `effective_from`) với thời điểm thông tin có thể được sử dụng (`available_at`). Pipeline chỉ cho phép một fold sử dụng bản ghi có `available_at` không muộn hơn thời điểm ra quyết định. `fetched_at` là thời điểm hệ thống lấy dữ liệu và không thay thế `available_at`.

Mỗi bảng nghiên cứu phải có `source`, `source_url`, `fetched_at` và `raw_checksum`. Security master và membership history phải có `history_method`. Các giá trị được chấp nhận cho research mode là lịch sử sự kiện chính thức, lịch sử niêm yết từ sở giao dịch hoặc lịch sử membership đã xác minh. Phương pháp suy ngày niêm yết từ phiên giá đầu tiên mang nhãn `first_price_observation_proxy` và bị leakage audit từ chối.

## Universe lịch sử

Universe tại mỗi kỳ được tạo theo một định nghĩa khai báo rõ. `hose_all_listed` dùng lịch sử niêm yết/hủy niêm yết của Sở; `index_membership` dùng các khoảng thành viên của đúng chỉ số được chọn. Hai khái niệm không được thay thế cho nhau. Snapshot lưu nguồn, checksum và lý do đủ điều kiện. Nếu thiếu provenance hoặc hợp đồng dữ liệu cốt lõi, research run tạo artifact `blocked` và dừng trước huấn luyện.

## Giá và sự kiện doanh nghiệp

Adapter phải giữ nguyên availability timestamp thay vì ghi đè bằng ngày quan sát. Dữ liệu giá điều chỉnh chỉ được dùng khi có hợp đồng giá điều chỉnh đã xác minh, hoặc giá thô được nối với bảng corporate action point-in-time. Hệ thống xuất từng outlier trên 30% vào `return_outlier_review.csv`; outlier không có cách giải thích được xác minh là lỗi data-quality, không còn là warning bỏ qua được.

## Dữ liệu tùy chọn

Báo cáo tài chính, vĩ mô và foreign flow chỉ tham gia feature set khi có coverage theo fold lớn hơn ngưỡng cấu hình và thỏa hợp đồng thời gian. Biến hoàn toàn thiếu bị loại trước imputation. Thiếu bảng tùy chọn phải xuất hiện trong limitations; hệ thống không điền dữ liệu giả trong research mode.

## Fail-closed

Research mode không dùng fixture, không dùng historical universe suy diễn và không tạo metrics khi audit bị chặn. Demo fixture vẫn chạy toàn bộ code path nhưng mọi artifact mang nhãn `NOT RESEARCH RESULT`.

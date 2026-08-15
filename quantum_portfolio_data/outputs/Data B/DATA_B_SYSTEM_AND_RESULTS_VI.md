# Mô tả hệ thống và kết quả nghiên cứu Data B

## 1. Thông tin nhận dạng kết quả

- Tên phiên bản: **Data B**.
- Experiment ID: `20260815T184821-a800b2584d`.
- Hash cấu hình: `a800b2584d`.
- Hash panel giá: `6e046b509fef366681866328d5bd99ec63541c2de8597f0e7bebc101813baa05`.
- Dữ liệu: 565.471 quan sát, 394 mã, từ 02/01/2020 đến 31/12/2025.
- Giai đoạn kiểm định ngoài mẫu thực tế: 02/05/2022–30/12/2025.
- Số fold: 44/44, theo tháng liên tục.
- Trạng thái: chạy thành công; kết quả mang tính khám phá, chưa phải bằng chứng confirmatory hay khuyến nghị đầu tư.

## 2. Giải thích hệ thống theo cách dễ hiểu

Hệ thống có thể được hiểu như một quy trình sàng lọc nhiều tầng. Từ gần 400 cổ phiếu, hệ thống chọn ra tập ứng viên nhỏ, dùng thuật toán lượng tử mô phỏng để chọn đúng bốn mã, sau đó dùng tối ưu cổ điển để chia tiền. Khi thị trường xấu hoặc không có đủ tín hiệu lợi nhuận dương, hệ thống giảm tỷ trọng cổ phiếu và giữ phần lớn tiền mặt.

Quy trình cụ thể gồm các bước sau:

1. **Khóa dữ liệu và nguồn gốc.** Data B không crawl lại mà tái sử dụng panel bất biến của Data A. Hash dữ liệu được kiểm tra trước khi chạy. Các nguồn giá trong panel gồm FinanceDataReader/Yahoo, Vnstock/KBS và các đoạn được đối chiếu hoặc thay thế bằng CafeF.
2. **Tạo đặc trưng.** Hệ thống tính các biến động lượng, xu hướng, biến động, downside risk, drawdown và thanh khoản chỉ từ dữ liệu đã xuất hiện trước ngày ra quyết định.
3. **Chia walk-forward.** Mỗi fold dùng 24 tháng huấn luyện, 3 tháng validation và 1 tháng test. Khoảng embargo 20 ngày được áp dụng để loại nhãn lợi nhuận tương lai có thể chồng lấn ranh giới thời gian.
4. **Xếp hạng cổ phiếu.** XGBoost học quan hệ giữa đặc trưng và thứ hạng lợi nhuận 20 ngày. Song song, một điểm kỹ thuật tổng hợp được tính từ momentum, trend và rủi ro. Tỷ lệ trộn XGBoost–kỹ thuật được chọn riêng trong validation của từng fold, không dùng test.
5. **Hiệu chỉnh điểm sang lợi nhuận kỳ vọng.** Điểm xếp hạng không được coi trực tiếp là phần trăm lợi nhuận. Hệ thống dùng dữ liệu validation đã purge để ánh xạ điểm về thang lợi nhuận 20 ngày.
6. **Adaptive Universe Reduction.** Cơ chế AUR kết hợp tín hiệu, thanh khoản, rủi ro, tương quan, độ ổn định giữa hai kỳ và ngân sách qubit. Trước AUR, mã không thể mua ít nhất một lô 100 cổ phiếu trong quy mô vốn và mức giải ngân tối thiểu bị loại. Kết quả thường là 8 ứng viên.
7. **Ước lượng rủi ro bằng EWMA covariance.** Đây là EWMA đa biến: đầu vào là ma trận lợi nhuận của nhiều cổ phiếu, đầu ra là vector trung bình và ma trận hiệp phương sai có trọng số giảm dần theo thời gian. Data B dùng span 60 và quy đổi sang chân trời 20 ngày.
8. **Lập QUBO.** Tín hiệu lợi nhuận đã hiệu chỉnh và ma trận EWMA covariance được kết hợp thành bài toán QUBO, yêu cầu chọn đúng 4 trong tối đa 8 mã.
9. **So sánh solver.** Cùng một instance được giải bằng exact solver, simulated annealing, penalty-QAOA và feasible-subspace XY-QAOA khởi tạo Dicke. XY mixer giữ nguyên Hamming weight nên luôn chọn đúng bốn mã trên ideal statevector. Danh mục triển khai dùng nghiệm khả thi tốt nhất đã quan sát trong các shot, không dùng bitstring có xác suất cao nhất nếu nó có giá trị mục tiêu kém hơn.
10. **Tối ưu tỷ trọng.** Sau khi chọn bốn mã, SLSQP phân bổ tỷ trọng long-only với cận trên, cận dưới, phạt turnover, giới hạn theo ADV và ràng buộc khả năng mua lô. Sector cap được khai báo nhưng không được áp dụng vì metadata ngành chưa đủ.
11. **Điều chỉnh mức giải ngân.** Median return của thị trường trong 63 và 126 ngày, cùng biến động 60 ngày, xác định mức giải ngân. Khi xu hướng xấu hoặc không có đủ bốn mã có lợi nhuận kỳ vọng dương, mức cổ phiếu bị giới hạn ở 25%; phần còn lại là tiền mặt.
12. **Mô phỏng giao dịch.** Danh mục được làm tròn theo lô 100, giả định quy mô 100 triệu đồng và tính commission, thuế bán, slippage, market impact, turnover và phần tiền mặt dư.
13. **Đánh giá.** Hệ thống tính lợi nhuận, volatility, Sharpe, Sortino, maximum drawdown, Calmar, chi phí; thực hiện ablation, sensitivity, block bootstrap và hiệu chỉnh Holm cho nhiều phép so sánh.

## 3. Kết quả Data B

Pipeline đầy đủ đạt lợi nhuận lũy kế sau chi phí **16,13%**, tương ứng lợi nhuận thường niên hóa **4,08%**. Volatility thường niên hóa là **7,54%**, Sharpe **0,176**, Sortino **0,175** và maximum drawdown **-14,05%**. Lợi nhuận gộp lũy kế đạt **20,75%**; tổng chi phí giao dịch ghi nhận theo tỷ trọng là **3,90%**, với tổng turnover một chiều **18,62** lần giá trị danh mục trong toàn bộ giai đoạn.

Trong 44 fold, 25 fold có lợi nhuận dương và 19 fold có lợi nhuận âm. Fold tốt nhất đạt 14,98%, trong khi fold kém nhất giảm 5,50%. Kết quả theo năm lần lượt là -10,97% trong phần năm 2022 được kiểm định, 20,34% năm 2023, 3,88% năm 2024 và 4,34% năm 2025.

So với các đối chứng, Data B cao hơn Equal Weight toàn universe (6,95%), Markowitz Mean–Variance (-31,24%) và adaptive exact (-8,93%), nhưng thấp hơn Minimum Variance (28,16%) và liquidity top-k exact (19,35%). Không có chênh lệch lợi nhuận nào giữa pipeline đầy đủ và các benchmark đạt ý nghĩa thống kê sau hiệu chỉnh Holm.

XGBoost có Rank IC trung bình 0,0599, trong khi Rank IC của tín hiệu tối ưu hóa sau trộn là 0,0534 và EWMA signal đối chứng là -0,0027. Chênh lệch XGBoost–EWMA chưa có ý nghĩa thống kê. Trên validation, hệ thống chọn trọng số XGBoost 1,00 trong 24 fold; 0,75 trong 7 fold; 0,50 trong 5 fold; và 0,25 trong 8 fold.

AUR không thể hiện ưu thế về lợi nhuận kỳ tiếp theo hoặc đa dạng hóa so với top-M cố định trong run này. Lợi nhuận tương lai trung bình của tập adaptive là -1,05%, so với -0,83% của fixed top-M; tương quan tuyệt đối trung bình tương ứng là 0,335 và 0,304. Do đó H2 không được hỗ trợ.

XY-QAOA đạt feasibility rate 100% trên cấu hình chính, cao hơn penalty-QAOA, và H3 là giả thuyết duy nhất được hỗ trợ thống kê. Nếu đánh giá bitstring khả thi có xác suất cao nhất, optimality gap trung bình của XY-QAOA là 3,40% trên toàn bộ run. Danh mục triển khai lại sử dụng nghiệm khả thi tốt nhất đã được quan sát; với ba seed và 1.024 shot, nghiệm này trùng giá trị exact trong cả 44 fold. Điều này không chứng minh quantum advantage, vì exact solver và simulated annealing cũng giải được các instance 7–8 biến rất nhanh.

## 4. Rổ cổ phiếu cuối cùng

Tại ngày ra quyết định 28/11/2025, hệ thống xác định trạng thái risk-off và đặt mức giải ngân mục tiêu 25%. Sau khi làm tròn lô 100 trên danh mục 100 triệu đồng, rổ thực thi gồm:

| Mã | Tên doanh nghiệp | Tỷ trọng thực thi | Số cổ phiếu | Lợi nhuận 20 ngày kỳ vọng của mã |
| --- | --- | ---: | ---: | ---: |
| SMB | CTCP Bia Sài Gòn - Miền Trung | 8,0400% | 200 | 0,7354% |
| KOS | CTCP KOSY | 3,8950% | 100 | 0,3955% |
| HTG | Tổng CTCP Dệt may Hòa Thọ | 3,8917% | 100 | 0,7167% |
| TLD | CTCP Đầu tư Xây dựng và Phát triển Đô thị Thăng Long | 3,2760% | 400 | 0,6495% |

Tổng tỷ trọng cổ phiếu thực thi là **19,10%** và tiền mặt thực tế là **80,90%**. Lợi nhuận 20 ngày kỳ vọng của toàn danh mục sau làm tròn lô là **0,1237%**. Đây là kỳ vọng ex-ante của mô hình, không phải lợi nhuận được bảo đảm. Lợi nhuận thực tế của fold tháng 12/2025 là **-0,2595%** sau chi phí.

## 5. Vì sao Data B chuyển từ âm sang dương?

Data B dương không phải vì dữ liệu giá được thay đổi; panel giá có cùng hash với Data A. Khác biệt chủ yếu nằm ở cơ chế quyết định và quản trị rủi ro:

- Tín hiệu XGBoost được trộn với factor kỹ thuật bằng validation thay vì dùng một cấu hình cố định cho mọi thời kỳ.
- AUR tăng vai trò của thanh khoản, áp dụng ngưỡng rủi ro, cụm tương quan và điều kiện khả năng giao dịch.
- Nghiệm triển khai là bitstring khả thi tốt nhất đã quan sát từ XY-QAOA, giúp tránh dùng nghiệm xác suất cao nhưng có energy kém.
- Ràng buộc lô 100, quy mô vốn, ADV và tiền mặt được đưa trực tiếp vào quy trình thực thi.
- Quan trọng nhất, exposure overlay giữ phần lớn vốn bằng tiền mặt. Mức giải ngân mục tiêu trung bình chỉ 30,57%, còn mức thực thi trung bình sau lô là 26,64%. Cơ chế này làm volatility và drawdown giảm mạnh.

Tuy nhiên, lợi nhuận dương chưa bền vững. Hai fold tháng 7 và 8/2023 đóng góp lớn, với lợi nhuận lần lượt 6,68% và 14,98%. Nếu loại hai fold này, các fold còn lại có lợi nhuận cộng dồn khoảng -5,33%. Vì vậy không thể kết luận rằng AI–Quantum là nguyên nhân duy nhất tạo ra lợi nhuận, hoặc hệ thống đã có alpha ổn định.

## 6. CHƯƠNG 4: KẾT QUẢ NGHIÊN CỨU

### 4.1. Kết quả kiểm tra dữ liệu và thiết kế thực nghiệm

Nghiên cứu được thực hiện trên bộ dữ liệu Data B gồm 565.471 quan sát của 394 mã cổ phiếu trong giai đoạn từ ngày 02/01/2020 đến ngày 31/12/2025. Kết quả kiểm tra chất lượng cho thấy panel giá đáp ứng các yêu cầu về khóa mã–ngày, cấu trúc trường giá, lớp provenance và phạm vi thời gian. Hash dữ liệu được cố định trong manifest nhằm bảo đảm khả năng truy vết và tái lập.

Quy trình walk-forward tạo ra 44 fold liên tục. Mỗi fold sử dụng 24 tháng huấn luyện, 3 tháng xác thực và 1 tháng kiểm định, kèm khoảng embargo 20 ngày. Do yêu cầu phải có đủ dữ liệu huấn luyện và xác thực, giai đoạn ngoài mẫu thực tế bắt đầu từ ngày 02/05/2022 và kết thúc vào ngày 30/12/2025. Toàn bộ tham số tiền xử lý, mô hình và tỷ lệ trộn tín hiệu được xác định từ training/validation; test window chỉ được sử dụng để đánh giá sau khi quyết định danh mục đã được khóa.

Mặc dù các kiểm tra point-in-time của pipeline đạt yêu cầu ở chế độ exploratory, dữ liệu vẫn chưa có hợp đồng điều chỉnh giá đã được xác minh đầy đủ và chưa có benchmark total-return đáng tin cậy. Do đó, kết quả trong chương này được diễn giải như bằng chứng khám phá thay vì kết luận xác nhận.

### 4.2. Kết quả mô hình xếp hạng và thu hẹp tập tài sản

Kết quả ngoài mẫu cho thấy XGBoost đạt Rank IC trung bình 0,0599. Dấu dương của chỉ tiêu này cho thấy các mã được mô hình xếp hạng cao có xu hướng đạt lợi nhuận tương đối tốt hơn trong kỳ tiếp theo. Tuy nhiên, Rank IC dao động từ -0,3899 đến 0,5617, phản ánh sự bất ổn định đáng kể giữa các trạng thái thị trường. Phép so sánh XGBoost với EWMA signal không đạt ý nghĩa thống kê sau hiệu chỉnh Holm; vì vậy H1 chưa được hỗ trợ thống kê.

Cơ chế adaptive universe reduction tạo tập 8 ứng viên trong 43 fold và 7 ứng viên trong một fold. Trung bình mỗi fold có khoảng 24 mã bị loại trước solver do không đáp ứng đồng thời điều kiện lô 100, quy mô vốn, mức giải ngân tối thiểu và trần tỷ trọng. So sánh với top-M cố định không cho thấy AUR cải thiện có ý nghĩa thống kê về lợi nhuận kỳ tiếp theo hoặc mức độ đa dạng hóa. Do đó, H2 chưa được hỗ trợ.

### 4.3. Kết quả các bộ giải

Exact solver và simulated annealing đều đạt feasibility rate 100% và optimality gap trung bình bằng 0 trên các instance có quy mô 7–8 biến. Penalty-QAOA đạt feasibility rate trung bình 56,95% và optimality gap 32,60% đối với nghiệm chính. Ngược lại, feasible-subspace XY-QAOA đạt feasibility rate 100%, xác nhận khả năng bảo toàn ràng buộc cardinality của Dicke state và XY mixer trong mô phỏng lý tưởng. Chênh lệch feasibility giữa hai biến thể QAOA có ý nghĩa thống kê sau hiệu chỉnh Holm, qua đó hỗ trợ H3.

Đối với XY-QAOA, optimality gap trung bình của bitstring khả thi có xác suất cao nhất là 3,40%. Khi sử dụng bitstring khả thi có energy tốt nhất đã được quan sát qua ba seed và 1.024 shot, gap của nghiệm triển khai bằng 0 trong 44 fold. Tuy nhiên, phép kiểm định chênh lệch optimality gap giữa XY-QAOA và penalty-QAOA không còn ý nghĩa sau hiệu chỉnh Holm. Do đó H4 chưa được hỗ trợ thống kê. Kết quả này chỉ phản ánh chất lượng nghiệm trên ideal statevector simulator, không phản ánh lợi thế tốc độ hoặc quantum advantage trên phần cứng thực.

### 4.4. Kết quả hiệu quả danh mục

Pipeline đầy đủ đạt lợi nhuận lũy kế sau chi phí 16,13%, lợi nhuận thường niên hóa 4,08%, volatility thường niên hóa 7,54% và maximum drawdown -14,05%. Sharpe ratio và Sortino ratio lần lượt là 0,176 và 0,175. Trong 44 fold, 25 fold có lợi nhuận dương, tương ứng tỷ lệ 56,82%.

Mặc dù pipeline đầy đủ vượt Equal Weight toàn universe và Markowitz Mean–Variance về lợi nhuận tích lũy, nó không vượt Minimum Variance và liquidity top-k exact. Block bootstrap cho thấy khoảng tin cậy của chênh lệch lợi nhuận hằng ngày so với các đối chứng đều chứa giá trị 0; toàn bộ p-value sau hiệu chỉnh Holm đều không có ý nghĩa. Do đó H5 chưa được hỗ trợ thống kê.

Mức rủi ro thấp của Data B chủ yếu gắn với exposure overlay. Trong 44 fold, có 25 fold được xếp loại risk-off, 7 fold mixed và 12 fold không có đủ bốn tài sản có lợi nhuận kỳ vọng dương. Mức giải ngân mục tiêu trung bình là 30,57%, trong khi tỷ trọng cổ phiếu thực thi trung bình sau làm tròn lô là 26,64%. Cơ chế giữ tiền mặt giúp giảm drawdown, nhưng đồng thời cho thấy kết quả dương không nên được quy hoàn toàn cho năng lực dự báo của AI hay bộ giải lượng tử.

### 4.5. Kết quả phân tích độ nhạy và kiểm định giả thuyết

Phân tích độ nhạy được thực hiện trên 660 trường hợp, thay đổi độ sâu QAOA, số shot, cardinality, seed và proxy nhiễu. Feasibility rate trong lưới độ nhạy dao động quanh 92–95%, trong khi optimality gap thay đổi theo cấu hình. Kết quả cho thấy thuật toán nhạy với tham số và không nên được đánh giá từ một cấu hình duy nhất. H6 được ghi nhận là đã hoàn thành phân tích sensitivity, không phải một kết luận vượt trội.

Tổng hợp kết quả kiểm định cho thấy chỉ H3 được hỗ trợ thống kê. H1, H2, H4 và H5 chưa được hỗ trợ; H6 được đánh giá thông qua lưới độ nhạy đã khai báo.

### 4.6. Thảo luận kết quả

Kết quả Data B cho thấy khung lai có thể vận hành end-to-end trên dữ liệu cổ phiếu Việt Nam, duy trì cardinality, sinh rổ cổ phiếu có khả năng giao dịch và tạo lợi nhuận lũy kế dương sau chi phí trong giai đoạn kiểm định. Đóng góp rõ nhất của thành phần lượng tử là duy trì không gian nghiệm khả thi, thay vì chứng minh ưu thế về lợi nhuận hoặc tốc độ. Đóng góp thực dụng rõ nhất của Data B là tích hợp được quy mô vốn, lô giao dịch, ADV, chi phí và tiền mặt vào cùng quy trình kiểm định.

Tuy nhiên, lợi nhuận của pipeline chưa thể hiện độ bền thống kê và phụ thuộc đáng kể vào hai fold tốt trong năm 2023. Minimum Variance vẫn có lợi nhuận và Sharpe cao hơn. Ngoài ra, dữ liệu chưa xác minh đầy đủ corporate actions/adjusted price, sector metadata còn thiếu, benchmark total-return chưa sẵn có và universe complete-case vẫn có nguy cơ coverage/survivorship bias. Vì vậy, nghiên cứu không khẳng định quantum advantage, không khẳng định alpha ổn định và không coi rổ cuối là khuyến nghị giao dịch.

### Kết luận Chương 4

Chương này đã trình bày kết quả thực nghiệm của khung tối ưu hóa danh mục lai AI–Quantum trên Data B. Pipeline đầy đủ đạt lợi nhuận sau chi phí dương và mức sụt giảm thấp hơn nhiều cấu hình đối chứng, trong khi XY-QAOA duy trì ràng buộc cardinality tốt hơn penalty-QAOA. Tuy vậy, pipeline chưa vượt Minimum Variance, các chênh lệch hiệu quả đầu tư chưa có ý nghĩa thống kê và lợi nhuận còn tập trung vào một số thời kỳ. Do đó, bằng chứng hiện tại hỗ trợ tính khả thi của kiến trúc và cơ chế bảo toàn nghiệm khả thi, nhưng chưa đủ để kết luận về ưu thế hiệu quả tài chính hay lợi thế lượng tử.

## 7. Hạn chế bắt buộc phải giữ trong báo cáo

1. Data B dùng panel complete-case và chưa có lịch sử thành viên HOSE point-in-time hoàn chỉnh, nên chưa loại trừ triệt để survivorship/coverage bias.
2. Chính sách adjusted price và corporate actions chưa được xác minh đầy đủ; kết quả chỉ được gắn nhãn exploratory.
3. Không có benchmark total-return đã xác minh; Equal Weight, Markowitz và Minimum Variance là benchmark nội bộ trên cùng panel.
4. Metadata ngành không đủ nên sector cap chưa thực thi trong các fold chính.
5. XY-QAOA chạy trên ideal statevector simulator, không phải phần cứng NISQ thực.
6. Lợi nhuận dương chưa có ý nghĩa thống kê và tập trung vào hai fold tốt; không được diễn giải là alpha ổn định.
7. Data B được phát triển sau khi đã quan sát kết quả Data A, do đó vẫn phải được xác nhận trên một holdout mới hoặc dữ liệu tương lai chưa dùng trong phát triển.

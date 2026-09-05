# Báo cáo thử nghiệm ràng buộc và tìm kiếm cấu hình

## 1. Mục tiêu và nguyên tắc

Mục tiêu của thử nghiệm là khắc phục mức tập trung cao và hiệu quả đầu tư kém của cấu hình ban đầu, đồng thời kiểm tra lại các giả thuyết về AUR và QAUR. Mọi cấu hình đều giữ nguyên nguyên tắc so sánh có kiểm soát: AUR và QAUR dùng cùng input tín hiệu, cùng risk estimator và cùng downstream portfolio pipeline.

Việc chọn cấu hình chỉ dùng 29 development folds đầu. Mười lăm folds cuối được dùng làm temporal holdout. Selection score là Sharpe trung bình của AUR và QAUR trên development, do đó không ưu tiên riêng một reducer.

## 2. Phạm vi thử nghiệm

- 157.826 price observations, 120 cổ phiếu, giai đoạn 2020–2025.
- 44 walk-forward folds với 916 phiên out-of-sample.
- Development: folds 0–28, 604 phiên.
- Temporal holdout: folds 29–43, 312 phiên.
- Chi phí giao dịch: 25 bps.
- 43 lượt cấu hình đã chạy, tương ứng 42 specification khác nhau do một no-overlay control được lặp lại ở vòng audit.

Ba nhóm thử nghiệm bao gồm:

1. Ràng buộc và phân bổ: thay đổi \(K\), \(k_p\), trần/sàn tỷ trọng, equal weight, inverse volatility, normalized mean–variance, covariance shrinkage và turnover penalty.
2. Universe reduction: thay signal blend, correlation penalty, stability penalty, quy mô candidate set và warm-start QAUR.
3. Risk overlay dùng chung: validation-IC filter, market-regime filter 60/120/200 phiên và volatility target 15%.

Grid screening dùng exact enumeration trong fixed-cardinality feasible subspace. Cấu hình được chọn sau đó được audit lại trên 15 holdout folds bằng ideal fixed-Hamming-weight XY-QAOA statevector.

## 3. Cấu hình đề xuất

Cấu hình được chọn theo development score là `W_K10P6_CP30`:

- Candidate set: \(K=10\).
- Final portfolio: \(k_p=6\).
- Tỷ trọng tối thiểu 2%, tối đa 25%.
- Tín hiệu: 70% XGBoost rank và 30% momentum rank.
- Correlation penalty của \(Q^{UR}\): 0,30.
- Stability weight: 0 trong controlled solver comparison.
- Covariance: EWMA span 60 kết hợp 20% diagonal shrinkage.
- Weight optimizer: normalized mean–variance, risk aversion 2,0, turnover penalty 0,15.
- QAUR: multi-start cardinality-preserving local search, warm-start từ nghiệm khả thi AUR.
- Risk overlay: không dùng; các overlay thử nghiệm đều làm giảm development score.

Warm-start là một hybrid algorithm hợp lệ: QAUR nhận nghiệm AUR làm một điểm khởi tạo và chỉ giữ nguyên hoặc cải thiện \(Q^{UR}\). Do đó H1 trong cấu hình này phản ánh khả năng cải thiện của joint local search so với greedy feasible solution; nó không phải bằng chứng quantum advantage.

## 4. So sánh với cấu hình ban đầu trên toàn bộ 916 phiên

| Cấu hình | Method | Cumulative return | Annualized return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| Ban đầu | AUR | -16,24% | -4,76% | -0,254 | -57,56% |
| Ban đầu | QAUR | -9,92% | -2,83% | -0,150 | -57,56% |
| Đề xuất | AUR | +17,35% | +4,50% | 0,288 | -37,12% |
| Đề xuất | QAUR | +33,08% | +8,18% | 0,514 | -35,47% |

Kết quả này cho thấy việc sửa cardinality, weight cap, thang đo mean–variance, shrinkage, turnover penalty và signal blend đã khắc phục đáng kể cấu hình ban đầu trong cùng local runtime. Bảng toàn mẫu chỉ mang tính mô tả; kết luận đánh giá cấu hình phải dựa trên holdout.

## 5. Kết quả temporal holdout

| Method | Cumulative return | Annualized return | Volatility | Sharpe | Sortino | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|
| AUR | +24,81% | +19,60% | 14,27% | 1,374 | 1,761 | -12,44% |
| QAUR | +23,51% | +18,59% | 13,90% | 1,338 | 1,734 | -12,82% |
| Full-Universe EW | +24,41% | +19,29% | 13,94% | 1,384 | 1,390 | -17,46% |
| VNAllshare TRI | +40,01% | +31,24% | 19,93% | 1,567 | 1,747 | -18,34% |

Cả hai nhánh đề xuất đều dương mạnh và giảm drawdown so với hai baseline. Tuy nhiên, QAUR không tạo return cao hơn AUR trong final holdout; cả hai cũng không vượt VNAllshare TRI về return hoặc Sharpe. Full-Universe EW có Sharpe nhỉnh hơn hai nhánh, mặc dù drawdown sâu hơn.

`W_K8P4_CP50` cho kết quả QAUR holdout +38,16% và Sharpe 1,970, nhưng cấu hình này có development Sharpe âm và không được chọn theo selection rule. Vì vậy, nó chỉ là exploratory candidate cho nghiên cứu tiếp theo, không phải đề xuất confirmatory.

## 6. Kiểm định giả thuyết trên holdout

| Giả thuyết | Estimate QAUR − AUR | One-sided p-value | Kết luận 5% |
|---|---:|---:|---|
| H1: QAUR có \(Q^{UR}\) objective cao hơn | +0,060278 | 0,000083 | Ủng hộ |
| H2: QAUR có candidate correlation thấp hơn | -0,012364 | 0,000004 | Ủng hộ |
| H3: candidate turnover của QAUR non-inferior, margin 2 điểm % | -0,060000 | 0,015841 | Ủng hộ |
| H4: QAUR có mean daily return cao hơn | -0,000036 | 0,567731 | Không ủng hộ |
| H5: khác biệt tài chí bền vững qua seed/configuration | QAUR Sharpe thấp hơn AUR ở 3/3 seed của cấu hình chọn | — | Không ủng hộ |

H5 cần được tách thành hai phần trong report:

- Structural robustness được ủng hộ: objective gap dương và correlation gap âm trong 6/6 warm-start specifications, ở cả development và holdout. Mean objective gap cũng dương trong 3/3 QAUR seeds.
- Financial superiority không được ủng hộ: dấu của Sharpe/return difference thay đổi theo configuration; trong cấu hình được chọn, QAUR holdout Sharpe thấp hơn AUR với cả ba seed.

Candidate turnover của QAUR thấp hơn AUR 6 điểm phần trăm trên holdout, nhưng downstream portfolio turnover lại cao hơn khoảng 2,10 điểm phần trăm. Do đó H3 chỉ được chấp nhận cho universe turnover, không được mở rộng sang portfolio turnover.

## 7. XY-QAOA audit

Trên 15 holdout folds và cả hai reducer:

- Feasibility rate trung bình: 100%.
- Optimality gap của best observed state: 0.
- Success probability trung bình: xấp xỉ 4%.

Kết quả này chỉ cho thấy fixed-Hamming-weight simulator bảo toàn cardinality và quan sát được exact-best state với 1.024 shots. QAUR vẫn dùng classical surrogate và XY-QAOA vẫn là ideal statevector; không có kết luận quantum advantage.

## 8. Rổ cổ phiếu ở fold cuối

Decision date là 31/10/2025; đây là historical out-of-sample result, không phải khuyến nghị giao dịch tại năm 2026.

- AUR Top-10: BWE, NAF, QNP, VCB, VCF, VID, VJC, VPD, VPI, VRE.
- AUR final portfolio: BWE 25%; VID 25%; VPD 25%; QNP 21%; VCB 2%; VCF 2%.
- QAUR Top-10: BWE, CIG, NAF, QNP, VCF, VID, VJC, VPD, VPI, VRE.
- QAUR final portfolio: BWE 25%; VID 25%; VPD 25%; VJC 14,58%; QNP 8,42%; CIG 2%.

## 9. Kết luận đề xuất

`W_K10P6_CP30` là đề xuất tốt nhất cho bản nghiên cứu hiện tại vì nó thỏa ba điều kiện cùng lúc: được chọn mà không nhìn holdout; chuyển kết quả toàn mẫu từ âm sang dương và giảm drawdown rõ rệt; đồng thời cung cấp bằng chứng holdout cho H1, H2 và H3.

Kết luận đúng không phải “QAUR sinh lời cao hơn AUR”. Kết luận đúng là: warm-start QAUR cải thiện \(Q^{UR}\) objective, giảm pairwise correlation và không làm tăng candidate turnover trong thử nghiệm này; tuy nhiên, các cải thiện ở universe-reduction layer chưa chuyển hóa thành lợi nhuận hoặc Sharpe cao hơn AUR trên final holdout. Framework đã tốt hơn rõ rệt nhưng chưa thể tuyên bố alpha so với VNAllshare TRI hoặc quantum advantage.

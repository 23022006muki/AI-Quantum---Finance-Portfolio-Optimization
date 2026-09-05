# Hướng cải thiện tốt nhất sau live-readiness và quantum benchmark

## Kết luận ngắn

Hướng tốt nhất hiện tại là **multi-lookback prequential QAUR ensemble**, không phải một static configuration duy nhất. Tại mỗi fold, năm selector chỉ dùng 6, 9, 12, 18 và 24 folds quá khứ để chọn một warm-start QAUR configuration; lợi nhuận của năm selector sau đó được chia đều. Không selector nào được phép nhìn current/future fold.

Trên 312 phiên holdout:

| Nhánh | Cumulative return | Annualized return | Volatility | Sharpe | Sortino | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Prequential AUR ensemble | +21,05% | +16,68% | 14,69% | 1,136 | 1,468 | -12,36% |
| Prequential QAUR ensemble | +29,59% | +23,29% | 14,13% | 1,648 | 2,203 | -12,53% |
| Full-Universe EW | +24,41% | +19,29% | 13,94% | 1,384 | 1,390 | -17,46% |
| VNAllshare TRI | +40,01% | +31,24% | 19,93% | 1,567 | 1,747 | -18,34% |

QAUR ensemble nhỉnh hơn Full-Universe EW về return, Sharpe và drawdown, đồng thời có Sharpe/Sortino cao hơn VNAllshare nhưng return thấp hơn. Kết quả QAUR cũng khá bền vững theo lookback: cumulative holdout nằm trong khoảng +24,49% đến +36,09%, Sharpe trong khoảng 1,339–1,820 với cả năm lookback.

## Vì sao vẫn chưa đủ để triển khai vốn thật quy mô lớn

Live-capital audit chỉ qua 4/9 gates:

- PASS: return holdout dương cho cả hai nhánh.
- PASS: static-config holdout Sharpe trên 1.
- PASS: maximum drawdown không sâu hơn -20%.
- PASS: static strategy vẫn có Sharpe dương tại 75 bps; tại 100 bps AUR Sharpe 1,016 và QAUR Sharpe 0,958.
- FAIL: chưa có excess return có ý nghĩa so với Full-Universe EW.
- FAIL: chưa có excess return có ý nghĩa so với VNAllshare TRI.
- FAIL: Deflated Sharpe Probability chưa đạt 95%.
- FAIL: chưa có tối thiểu 24 tháng untouched forward/paper track record.
- FAIL: dataset dừng ở 31/12/2025, chưa có audited 2026 data feed và order pipeline.

Với QAUR ensemble, annualized arithmetic excess return so với Full-Universe EW là +3,31%, nhưng 95% moving-block-bootstrap interval là [-19,02%; +20,94%] với one-sided p-value 0,4714. So với VNAllshare, chênh lệch là -7,25%/năm với p-value 0,7790. Deflated Sharpe Probability của ensemble QAUR là 0,9468 sau khi điều chỉ 44 trials, gần nhưng chưa đạt gate 0,95.

## Quantum benchmark cho kết quả gì

Trên sáu full-universe instances đại diện:

| Solver | Mean \(Q^{UR}\) objective | Mean runtime |
|---|---:|---:|
| Greedy AUR | 4,867532 | 0,0041 giây |
| Warm single-swap QAUR | 4,917950 | 0,0108 giây |
| Multistart-swap QAUR | 4,923384 | 0,1998 giây |
| Simulated annealing | 4,923384 | 1,8316 giây |

Multistart-swap QAUR khớp simulated annealing trên 6/6 instances và nhanh hơn khoảng chín lần. Đây là bằng chứng cho một **classical hybrid heuristic hiệu quả**, không phải quantum advantage.

Trên các small instances \(n=8,10,12\), XY-QAOA statevector quan sát được exact optimum với gap 0, nhưng exact enumeration, greedy, local search và simulated annealing cũng đạt gap 0. Success probability của XY-QAOA giảm xấp xỉ từ 5% ở \(n=8\) xuống 1% ở \(n=12\); simulator runtime tăng từ khoảng 0,05 lên 0,33 giây. Quantum-advantage audit hiện qua 0/7 gates.

## Lộ trình tốt nhất cho vốn thật

1. Freeze ngay rule của multi-lookback ensemble, warm-start QAUR, cost model và toàn bộ pass/fail gates; không tuning thêm trên 2020–2025.
2. Mở rộng point-in-time adjusted dataset từ 01/01/2026 đến hiện tại, giữ nguyên rule đã freeze. Khoảng dữ liệu này phải được xem là untouched forward test.
3. Chạy shadow portfolio tối thiểu 6–12 tháng với bid–ask spread, slippage, partial fills, limit/market order rule và capacity theo ADV.
4. Chỉ mở pilot 1–2% vốn khi forward DSR đạt 0,95, net Sharpe tại 75 bps trên 1, maximum drawdown không vượt 20% và không có data/execution breach.
5. Tăng lên 5–10% vốn sau tối thiểu 12 tháng paper + small-live track record. Chỉ xem xét full allocation sau 24 tháng forward evidence và significant net excess return so với benchmark phù hợp.
6. Bổ sung sector cap, ADV cap, liquidity floor, volatility targeting, kill switch, loss limit, stale-data detection và rollback portfolio trước khi giao dịch thật.

## Lộ trình duy nhất có thể dẫn đến quantum-advantage claim

1. Giữ tên hiện tại là `quantum-ready` hoặc `quantum-assisted formulation`; không gọi là quantum advantage.
2. Chạy cùng \(Q^{UR}\) instances trên physical QPU/quantum annealer, không chỉ statevector.
3. Benchmark với Gurobi/CPLEX/SCIP, tuned simulated annealing, tabu/local search, GPU/HPC và exact solver khi khả thi, với cùng time/energy budget.
4. Tính end-to-end wall clock bao gồm embedding, transpilation, queue, calibration, shots, error mitigation và classical parameter optimization.
5. Pre-register ba primary metrics: time-to-target, objective gap tại equal time budget và success probability; chạy tối thiểu 30 repeats trên real và synthetic instances theo nhiều kích thước.
6. Chỉ tuyên bố advantage nếu QPU có statistically significant quality/time/energy improvement và thể hiện scaling crossover trên một vùng kích thước bài toán, sau đó được nhóm độc lập tái lập.

## Đề xuất cuối

Phương án nên theo đuổi là **prequential multi-lookback QAUR ensemble + frozen rules + forward paper trading**. Đây là phương án có risk-adjusted holdout result tốt nhất và ít phụ thuộc vào một lookback duy nhất. Chưa được triển khai full capital và chưa được tuyên bố quantum advantage cho đến khi có forward-time evidence và physical-QPU benchmark.

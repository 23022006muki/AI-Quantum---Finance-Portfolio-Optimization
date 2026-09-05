# Đề xuất phương pháp tốt nhất sau vòng thử nghiệm mở rộng

## 1. Kết luận lựa chọn

Phương pháp được đề xuất cho giai đoạn paper trading tiếp theo là **C1-IV-X kết
hợp Common 30-Session Market Gate**. Đây là một framework phòng thủ theo regime,
trong đó AUR và QAUR vẫn là hai phương pháp universe reduction thay thế nhau;
toàn bộ portfolio-selection, weighting và risk overlay downstream được giữ giống
nhau giữa hai nhánh.

Phương pháp này được lựa chọn vì không tối đa hóa riêng một giai đoạn. Với chi
phí chuyển trạng thái 25 bps, lợi nhuận thấp nhất trong các tổ hợp
`giai đoạn × phương pháp reduction` vẫn đạt **+8,80%**, Sharpe thấp nhất đạt
**0,8714**, và maximum drawdown xấu nhất là **-15,77%**. Lookback 20, 30 và 40
phiên đều tạo kết quả dương trong development, historical holdout và observed
2026, cho thấy kết quả không phụ thuộc vào một điểm lookback duy nhất.

## 2. Cấu trúc phương pháp đề xuất

### Giai đoạn 1 — Forecasting

- XGBoost dự báo cross-sectional return rank.
- Không trộn momentum vào production signal trong protocol hiện tại.
- EWMA được sử dụng để ước lượng covariance/risk.
- Tất cả input tại thời điểm quyết định phải có timestamp và chỉ được sử dụng
  sau khi target tương ứng đã thực sự available.

### Giai đoạn 2 và 3 — Hai universe reducers

- **AUR:** lựa chọn tuần tự dựa trên unary score, liquidity, risk, stability và
  correlation redundancy.
- **QAUR:** tối ưu fixed-cardinality universe-reduction QUBO trên cùng input.
- Cả hai bắt đầu từ cùng full universe và tạo Top-8 candidate set.

### Giai đoạn 4 — So sánh universe reduction

So sánh AUR và QAUR bằng candidate objective, mean absolute correlation,
turnover, Jaccard similarity, downstream-selection overlap và stability. Không
được diễn giải XY-QAOA như đối thủ của AUR.

### Giai đoạn 5 — Shared portfolio pipeline

1. Cardinality-Constrained Portfolio QUBO chọn đúng 4 trong 8 ứng viên.
2. XY-QAOA là shared asset-selection solver cho cả hai nhánh.
3. Inverse-volatility allocation.
4. Bounded-simplex projection bảo đảm chính xác:

   \[
   \sum_i w_i=1,\qquad 0.05\leq w_i\leq0.30.
   \]

5. Chi phí giao dịch cơ sở 25 bps được trừ tại lần rebalance.

### Giai đoạn 5.5 — Common causal market gate

Tại mỗi thời điểm rebalance, tính equal-weight full-universe return của 30 phiên
trước đó:

\[
g_t=\prod_{s=t-30}^{t-1}(1+r_s^{EW})-1,
\qquad
e_t=\mathbb{1}[g_t>0].
\]

Tỷ trọng executable là:

\[
\widetilde{w}_{i,t}=e_t w_{i,t},
\qquad
w_{cash,t}=1-e_t.
\]

Gate này được áp dụng giống hệt cho AUR và QAUR, vì vậy không làm thay đổi đối
tượng so sánh. Khi regime chuyển giữa cash và risk-on, protocol tính thêm 25 bps
switching cost.

### Giai đoạn 6 — Prospective walk-forward paper test

- Rebalance theo tháng.
- Không thay tham số trong paper window.
- Ghi cả shadow portfolio và executable portfolio.
- Khi gate ở trạng thái cash, shadow portfolio vẫn được mark-to-market để tiếp
  tục đánh giá chất lượng AUR/QAUR và XY-QAOA.
- Mỗi thay đổi code, source policy hoặc parameter phải tạo protocol version mới.

## 3. Kết quả của phương pháp đề xuất

| Giai đoạn | AUR return | QAUR return | AUR Sharpe | QAUR Sharpe | MDD xấu nhất |
|---|---:|---:|---:|---:|---:|
| Development | 23,63% | 23,63% | 0,9814 | 0,9814 | -15,77% |
| Historical holdout | 14,43% | 19,57% | 0,8714 | 1,1389 | -13,10% |
| Observed 2026 | 8,81% | 8,80% | 1,1550 | 1,1540 | -7,78% |

Các số liệu trên đã bao gồm 25 bps chi phí chuyển trạng thái của overlay, ngoài
transaction cost nằm trong base portfolio return.

## 4. Trạng thái tháng 9/2026

Full-universe equal-weight return 30 phiên tại ngày khóa protocol là **-1,1352%**,
do đó regime tháng 9 là **CASH**.

Shadow portfolio của cả AUR và QAUR:

| Ticker | Shadow weight | Executable weight |
|---|---:|---:|
| NAF | 30,00% | 0,00% |
| VCB | 24,69% | 0,00% |
| VJC | 24,24% | 0,00% |
| STB | 21,07% | 0,00% |
| CASH | 0,00% | 100,00% |

## 5. Phạm vi kết luận

Kết quả cho phép kết luận rằng common market gate làm giảm regime dependence và
cải thiện độ ổn định lịch sử của base candidate. Tuy nhiên, overlay được thiết kế
sau khi đã quan sát dữ liệu 2026; vì vậy nó chỉ là phương pháp được đề xuất cho
giai đoạn tương lai, chưa phải bằng chứng triển khai vốn thật.

Framework vẫn chưa chứng minh quantum advantage. QAUR hiện dùng classical
cardinality-preserving search và XY-QAOA chưa được benchmark end-to-end trên QPU
thật. Mọi tuyên bố quantum advantage phải chờ physical-QPU scaling study với
exact solver, simulated annealing và classical multi-start làm đối chứng.

## 6. Điều kiện nâng cấp lên vốn thật

- Tối thiểu 12 tháng prospective paper record trước pilot vốn nhỏ.
- Tối thiểu 24 tháng trước đánh giá triển khai đầy đủ.
- Net return dương sau chi phí thực đo.
- MDD không thấp hơn -20%.
- Deflated Sharpe probability tối thiểu 95%.
- Excess return có p-value dưới 5% trên benchmark đã định trước.
- Giá, corporate action và executed order được cross-source audit.
- Không có unexplained data break hoặc order reconciliation break.


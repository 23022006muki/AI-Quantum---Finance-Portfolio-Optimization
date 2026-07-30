# Bộ prompt triển khai hệ thống AI–Quantum Portfolio

Tài liệu này gồm ba prompt chạy tuần tự trong cùng workspace. Mỗi prompt yêu cầu Codex
kiểm tra hiện trạng trước khi sửa mã nguồn, tái sử dụng sản phẩm của prompt trước và lưu
đầy đủ bằng chứng thực thi.

---

## Prompt 1 — Crawl và xây dựng lớp dữ liệu point-in-time

```text
Bạn là lead data engineer kiêm quantitative researcher. Hãy triển khai hoàn chỉnh đầu
việc thu thập dữ liệu cho đề tài tối ưu hóa danh mục đầu tư AI–Quantum trong workspace
hiện tại.

NGỮ CẢNH BẮT BUỘC
- Đọc toàn bộ cấu trúc workspace trước khi sửa. Đặc biệt đọc:
  - docs/
  - quantum_portfolio_data/
  - Proposal.pdf
  - C:\Users\LENOVO\Downloads\Data chỉnh sửa.pdf
  - C:\Users\LENOVO\Downloads\Proposal chỉnh sửa.pdf
  - C:\Users\LENOVO\Downloads\Báo cáo AI Quantum.pdf
  - C:\Users\LENOVO\Downloads\ThayDa_NhanXet.docx
- Nếu có mã nguồn/dữ liệu sẵn thì audit và mở rộng, không tạo project trùng lặp.
- Phạm vi nghiên cứu chính: cổ phiếu HOSE, ưu tiên giai đoạn 2015–2025.
- Đóng góp nghiên cứu trung tâm về sau là adaptive universe reduction theo qubit budget
  và correlation structure; dữ liệu phải hỗ trợ được thí nghiệm đó.
- Mọi dữ liệu dùng để ra quyết định tại ngày t chỉ được chứa thông tin thực sự công khai
  trước hoặc tại t.

MỤC TIÊU
Xây dựng pipeline crawl có thể chạy lại, tiếp tục sau lỗi, kiểm tra được nguồn gốc và tạo
ra bộ dữ liệu point-in-time. Ưu tiên hoàn thành Stage 1 đủ để chạy pipeline nghiên cứu;
Stage 2 và Stage 3 triển khai theo adapter/config và chỉ bật khi nguồn hợp lệ sẵn có.

PHẠM VI DỮ LIỆU
1. Stage 1 — bắt buộc:
   - Security master: ticker, company_name, exchange, industry, sector, listing_date,
     delisting_date nếu có, trạng thái giao dịch theo thời gian.
   - OHLCV hằng ngày: open, high, low, close, adjusted_close, volume, trading_value;
     bổ sung reference_price, market_cap, shares_outstanding khi nguồn cho phép.
   - Corporate actions: cash dividend, stock dividend, split/merge, bonus shares,
     rights issue, ex_date, announcement/source timestamp và adjustment factor.
   - Benchmark: VN-Index; VN30 và thành phần VN30 lịch sử nếu có nguồn đáng tin cậy.
   - Risk-free rate với release/effective timestamp rõ ràng.
2. Stage 2 — adapter/config:
   - Báo cáo tài chính và ngày công bố; tuyệt đối không join theo ngày kết thúc quý.
   - Các chỉ báo kỹ thuật không crawl; phải tính từ OHLCV ở pipeline sau.
3. Stage 3 — adapter/config:
   - Macro Việt Nam, USD/VND, dữ liệu quốc tế, market breadth và foreign flow.
   - Sentiment là tùy chọn, không được làm chậm Stage 1.

YÊU CẦU VỀ NGUỒN
- Ưu tiên nguồn chính thức, API công khai hoặc nguồn mà workspace đã cấu hình.
- Không hard-code cookie/token/secret. Tạo .env.example và config nguồn.
- Không vượt paywall, CAPTCHA, robots.txt hoặc điều khoản sử dụng.
- Không đoán endpoint. Nếu endpoint thật không truy cập được, tạo adapter rõ ràng,
  fixture nhỏ để test parser và báo blocker; không giả mạo dữ liệu là dữ liệu thật.
- Mỗi record/batch phải có provenance tối thiểu: source, source_url hoặc dataset id,
  fetched_at, available_at/release_at nếu có, checksum/raw artifact và parser version.

KIẾN TRÚC DỮ LIỆU
- Tách các lớp raw/bronze, normalized/silver, curated/gold.
- Raw là bất biến, lưu response gốc và manifest. Hỗ trợ incremental, retry với backoff,
  rate limit, cache, timeout, resume/checkpoint và idempotency.
- Chuẩn hóa timezone Asia/Ho_Chi_Minh, lịch giao dịch, mã cổ phiếu và đơn vị tiền tệ.
- Dùng định dạng cột hiệu quả (ưu tiên Parquet) và partition hợp lý theo source/date
  hoặc year/ticker; không tạo hàng nghìn file siêu nhỏ nếu tránh được.
- Định nghĩa schema/data contract cho security master, prices, corporate actions,
  index membership, benchmark và release-calendar data.
- Tái tạo universe U_t theo từng ngày tái cân bằng từ listing/delisting/trading status;
  không dùng danh sách cổ phiếu hiện tại để hồi cứu quá khứ.
- Nếu dùng VN30, chỉ dùng thành phần lịch sử có effective_from/effective_to.
- Không double-adjust giá. Ghi rõ adjusted_close do nguồn cung cấp hay do code tính;
  kiểm tra cổ tức, quyền mua, thưởng, split/merge quanh ex-date.

CHỐNG LEAKAGE VÀ DATA QUALITY
- Tạo validation tự động cho: uniqueness (date,ticker), schema/dtype, OHLC logic,
  volume/value không âm, duplicate, missingness, stale prices, trading-calendar gaps,
  outlier flags, corporate-action discontinuities và universe membership.
- Outlier chỉ được flag ở lớp dữ liệu; không winsorize toàn bộ lịch sử tại đây.
- Giữ riêng event_date, announcement_date, effective_date, available_at.
- Tạo leakage audit có test cho ít nhất:
  1) current-universe survivorship bias;
  2) VN30 membership lịch sử;
  3) corporate-action adjustment;
  4) financial statement availability date;
  5) dữ liệu có available_at > decision_time.
- Sinh báo cáo coverage theo ticker/năm/field/source và danh sách lỗi có mức độ.

CLI VÀ CẤU HÌNH KỲ VỌNG
Cung cấp CLI tương đương (được điều chỉnh theo codebase hiện có):
- crawl --stage 1 --from 2015-01-01 --to 2025-12-31
- normalize --stage 1
- validate --stage 1
- build-universe --rebalance monthly
- report-coverage --stage 1
Mọi command phải có --help, logging có cấu trúc, exit code đúng và dry-run nếu phù hợp.

TEST VÀ BẰNG CHỨNG
- Unit test parser/schema/adjustment/universe; integration test dùng fixture; một smoke
  test end-to-end trên khoảng thời gian và số ticker nhỏ.
- Chạy formatter/linter/type-check/test phù hợp với project.
- Chạy thử pipeline thật trong phạm vi an toàn. Nếu mạng/nguồn chặn, chạy fixture smoke
  test và ghi rõ phần nào chưa được xác minh bằng dữ liệu thật.
- Không được kết luận “đã crawl đủ 2015–2025” nếu coverage report không chứng minh.

DELIVERABLE
1. Mã nguồn và config có thể tái lập.
2. README hướng dẫn cài đặt, nguồn, license/ToS, CLI, schema, lịch cập nhật và recovery.
3. Data dictionary và source registry.
4. Báo cáo coverage + quality + leakage audit dạng máy đọc được và Markdown.
5. Dữ liệu mẫu/fixture không chứa secret.
6. Tệp RUN_REPORT.md ghi chính xác command đã chạy, trạng thái, thời gian, số bản ghi,
   coverage, lỗi còn lại và đường dẫn output.

CÁCH LÀM VIỆC
- Trước tiên inventory và viết ngắn gọn kế hoạch dựa trên code hiện có.
- Sau đó triển khai thật, không dừng ở kế hoạch hoặc skeleton.
- Không xóa/ghi đè thay đổi không liên quan của người dùng.
- Khi thiếu một nguồn, tiếp tục hoàn thiện các phần độc lập và ghi blocker cụ thể.
- Cuối cùng tóm tắt file đã đổi, command/test đã chạy, kết quả thật và các giới hạn.
```

---

## Prompt 2 — Xây dựng pipeline nghiên cứu và tối ưu hóa

```text
Bạn là principal ML/quantum engineer. Hãy xây dựng pipeline nghiên cứu hoàn chỉnh trên
lớp dữ liệu point-in-time đã tạo ở Prompt 1. Làm việc trực tiếp trong workspace hiện tại,
audit và tái sử dụng code/dữ liệu sẵn có; không tạo một project song song.

ĐỌC TRƯỚC KHI CODE
- Đọc docs/, Proposal.pdf, các PDF/DOCX trong C:\Users\LENOVO\Downloads đã nêu ở
  Prompt 1, README/data contracts/RUN_REPORT của pipeline dữ liệu và các bài báo liên
  quan trực tiếp trong docs/.
- Lấy nhận xét của giảng viên làm ràng buộc thiết kế:
  XGBoost hoặc EWMA tiền chọn tài sản
  → adaptive universe reduction theo qubit budget và correlation
  → cardinality-constrained QUBO
  → XY-QAOA + Dicke initialization
  → tối ưu tỷ trọng cổ điển
  → walk-forward backtest.
- Không mở rộng dàn trải sang VQE, Quantum Walk hay nhiều công nghệ khác trong phiên
  bản chính. Chúng chỉ có thể là tài liệu tham khảo hoặc future work.

MỤC TIÊU NGHIÊN CỨU
Triển khai pipeline có thể kiểm định sáu giả thuyết H1–H6 trong Proposal chỉnh sửa:
H1 chất lượng tiền chọn XGBoost; H2 giá trị của correlation-aware adaptive reduction;
H3 feasibility rate của XY-QAOA; H4 optimality gap/near-optimal probability; H5 hiệu
quả ngoài mẫu sau chi phí; H6 độ nhạy theo budget/depth/noise/market regime.
Đóng góp trung tâm cần được cô lập bằng ablation, không chỉ báo cáo lợi nhuận cuối.

PIPELINE BẮT BUỘC
1. Point-in-time universe:
   - Tạo U_t ở mỗi ngày tái cân bằng bằng listing/delisting/trading status/liquidity chỉ
     từ thông tin available_at <= t.
2. Walk-forward splits:
   - Mặc định train 36 tháng, validation 6 tháng, test 1 tháng, rebalance tháng.
   - Cho phép config fallback 24/3/1.
   - Không random split.
3. Preprocessing trong từng fold:
   - Imputer, outlier thresholds, scaler, feature selection chỉ fit trên training.
   - Validation chỉ để tuning; test tuyệt đối không dùng để chọn model/hyperparameter.
4. Feature engineering:
   - Tính từ dữ liệu đến t: return 5/20/60/120, SMA/EMA, RSI, MACD, ATR, volatility,
     downside volatility, beta, drawdown, liquidity/turnover.
   - Mỗi feature có lookback, lag và available_at; test chống dùng dữ liệu tương lai.
5. Target và ranking:
   - Ưu tiên cross-sectional rank của forward return kỳ nắm giữ.
   - XGBoost là model chính; EWMA, momentum và liquidity là baseline.
   - Báo IC, rank IC, precision/top-M return và độ ổn định theo fold.
6. Adaptive universe reduction:
   - Score A_i,t = w1*Z(signal) + w2*Z(liquidity) - w3*Z(risk).
   - Chọn M_t thỏa K <= M_t <= qubit budget B_t.
   - Penalize/kiểm soát tương quan; bảo đảm đại diện cụm/ngành nếu cấu hình bật.
   - So sánh với fixed Top-M, liquidity-only và random.
   - Lưu trace giải thích vì sao từng tài sản được chọn/loại.
7. Estimation:
   - Expected return và covariance chỉ dùng training history trước t.
   - So sánh sample covariance với Ledoit-Wolf/shrinkage; tùy chọn EWMA.
8. Cardinality-constrained QUBO:
   - Mục tiêu lựa chọn K tài sản cân bằng expected return và covariance.
   - Xây Q matrix/Hamiltonian có kiểm thử bằng brute-force trên instance nhỏ.
   - Tách rõ QUBO economic objective và penalty-QAOA baseline.
9. Solvers:
   - Exact/brute-force hoặc MILP phù hợp kích thước nhỏ làm ground truth.
   - Simulated annealing/classical heuristic.
   - Penalty-based QAOA baseline.
   - Feasible-subspace XY-QAOA với Dicke initial state là phương pháp chính.
   - Dùng simulator mặc định để tái lập; hardware backend là adapter tùy chọn.
   - Log seed, K, M, p, shots, optimizer, iterations, backend, noise, circuit depth,
     two-qubit gates, runtime, energy distribution và bitstring counts.
10. Classical weight optimization:
   - Trên tập tài sản đã chọn, giải mean-variance hoặc minimum variance.
   - Ràng buộc sum(w)=1, long-only, lower/upper bounds, turnover L1 và transaction cost.
11. Walk-forward backtest:
   - Signal dùng dữ liệu đến t; giao dịch sớm nhất ở phiên t+1.
   - Giữ trong test window, trừ transaction cost, dịch cửa sổ và lặp.
   - Không tuning dựa trên backtest cuối.
12. Evaluation:
   - Financial: cumulative/annualized return, volatility, Sharpe, Sortino, max drawdown,
     Calmar, turnover, costs, positive-month ratio.
   - Optimization: objective, optimality gap, approximation ratio nếu hợp lệ, exact/near
     optimum probability, runtime, stability across seeds.
   - Quantum: feasibility rate, energy, qubits, depth, two-qubit gates, shots, evaluations,
     noise/penalty sensitivity.
   - Universe: signal quality, average correlation, clusters/sectors represented,
     volatility và oracle/exact-set coverage chỉ dùng cho phân tích hậu nghiệm.

BENCHMARK VÀ ABLATION
- Financial: VN-Index, equal weight, minimum variance, mean-variance, risk parity nếu có.
- Selection: random, liquidity, momentum, EWMA, XGBoost Top-M, adaptive reduction.
- Solver: exact, simulated annealing, penalty-QAOA, XY-QAOA.
- Tối thiểu các cấu hình:
  1) no-AI/no-quantum;
  2) EWMA + classical;
  3) XGBoost + classical;
  4) adaptive reduction + classical;
  5) XGBoost + penalty-QAOA;
  6) XGBoost + XY-QAOA;
  7) adaptive reduction + XY-QAOA;
  8) full pipeline + weights + costs.
- Cùng input instance và compute budget khi so sánh solver; không cherry-pick seed.

THỐNG KÊ VÀ ĐỘ BỀN
- Block bootstrap confidence interval cho Sharpe/chênh lệch hiệu quả.
- Multiple-comparison correction khi cần.
- Sensitivity theo K, B/M, p, shots, noise, transaction cost, rebalance frequency,
  training length và bull/bear/sideway regime.
- Nếu compute quá lớn, tạo cấu hình quick/demo và full/research; không thay kết quả full
  bằng kết quả demo mà không gắn nhãn.

KIẾN TRÚC VÀ REPRODUCIBILITY
- Module hóa data access, features, splits, models, reduction, QUBO, solvers, weights,
  backtest, metrics, experiments và reporting.
- Config versioned; deterministic seeds; experiment ID; lưu dataset/config/code hash.
- Cache artifact tốn thời gian và resume thí nghiệm.
- Data contract rõ giữa các stage; output tối thiểu gồm panel features, fold manifest,
  rankings, selected universe, optimization instances, solver logs, weights, trades,
  returns, metrics và statistical tests.
- CLI tương đương:
  validate-data; build-features; make-folds; train-ranker; build-instances;
  run-solvers; optimize-weights; backtest; evaluate; run-experiment.

QUALITY GATES
- Test chống leakage cho mọi transformer và feature.
- Property tests: QUBO objective khớp công thức; XY mixer bảo toàn Hamming weight K;
  Dicke state chỉ có trạng thái khả thi; weights sum 1 và nằm trong bounds; trades dùng
  giá sau decision time.
- Unit/integration/end-to-end smoke test trên instance nhỏ.
- Chạy formatter/linter/type-check/test và một quick experiment thật.
- Không bịa quantum advantage. Nếu XY-QAOA không thắng baseline, báo đúng kết quả.
- Không gọi simulator là phần cứng lượng tử thật.

DELIVERABLE
1. Pipeline chạy được bằng một command cho quick experiment.
2. Config quick/demo và full/research.
3. Tests và leakage audit.
4. Experiment manifest + artifact có schema.
5. README kiến trúc, công thức, cách chạy, cách thêm backend.
6. RUN_REPORT.md ghi command, môi trường, data coverage, số fold, runtime, test results,
   metric thật và mọi giới hạn/blocker.

Hãy bắt đầu bằng audit hiện trạng và kế hoạch ngắn, sau đó triển khai, chạy và sửa lỗi đến
khi quick experiment hoàn tất. Không dừng ở pseudocode/skeleton. Bảo toàn thay đổi không
liên quan của người dùng.
```

---

## Prompt 3 — Hoàn thiện hệ thống, chạy demo và xuất kết quả nghiên cứu

```text
Bạn là staff software engineer kiêm reproducibility lead cho nghiên cứu. Hãy hoàn thiện
toàn bộ hệ thống AI–Quantum Portfolio dựa trên mã nguồn, dữ liệu, pipeline và tài liệu
trong workspace; chạy một demo end-to-end; xuất kết quả có thể kiểm chứng.

PHẠM VI VÀ NGUYÊN TẮC
- Đọc kết quả/RUN_REPORT của Prompt 1 và 2, toàn bộ docs/, Proposal.pdf, các tài liệu
  chỉnh sửa trong C:\Users\LENOVO\Downloads và các paper liên quan trực tiếp.
- Không viết lại pipeline đang hoạt động. Audit, tích hợp, bổ sung phần thiếu và sửa lỗi.
- Kiến trúc khoa học cố định:
  point-in-time data → XGBoost/EWMA → adaptive universe reduction
  → cardinality QUBO → XY-QAOA/Dicke (+ baselines)
  → classical weights → walk-forward backtest → evaluation.
- Đóng góp trung tâm là adaptive universe reduction + feasible-subspace QAOA.
- Không bịa dữ liệu, số liệu, citation, kết quả hoặc quantum advantage.

MỤC TIÊU SẢN PHẨM
Tạo một hệ thống có hai chế độ:
1. Reproducible research CLI: chạy experiment từ config đến report.
2. Demo UI cục bộ, ưu tiên Streamlit nếu codebase chưa có frontend, cho phép người dùng
   xem dữ liệu, cấu hình quick run, chạy/đọc kết quả và so sánh phương pháp.
UI không được chứa logic nghiên cứu cốt lõi; chỉ gọi service/pipeline đã kiểm thử.

TRƯỚC KHI TRIỂN KHAI
- Inventory file, dependency, config, test, artifact và git/worktree state.
- Chạy test và quick pipeline hiện có để tạo baseline.
- Lập ma trận “đã có / thiếu / lỗi / sẽ làm” cho data, ML, QUBO, solvers, optimization,
  backtest, reporting và demo.
- Nếu dữ liệu thật chưa đủ, không được âm thầm thay bằng synthetic. Có thể:
  a) chạy real-data subset đủ provenance; hoặc
  b) chạy fixture/synthetic smoke demo được gắn nhãn nổi bật “NOT RESEARCH RESULT”.

TÍCH HỢP HỆ THỐNG
- Một config duy nhất điều khiển data snapshot, date range, folds, features, model,
  qubit budget, M, K, QAOA p/shots/noise/seeds, constraints, costs và benchmarks.
- Preflight kiểm tra schema, coverage, point-in-time timestamps, dependency và resource
  budget trước khi chạy.
- Orchestrator hỗ trợ resume/cache và tạo experiment_id.
- Mỗi run lưu immutable manifest: code hash, config hash, dataset hash, dependency
  versions, seeds, start/end time, status và artifact index.
- Tách quick demo khỏi full research. UI phải cảnh báo rõ khi cấu hình quá tốn tài nguyên.

DEMO UI TỐI THIỂU
1. Overview: câu hỏi nghiên cứu, pipeline và trạng thái dữ liệu.
2. Data Quality: coverage, missingness, provenance, corporate actions, universe theo t,
   leakage-audit status.
3. Ranking & Reduction: XGBoost/EWMA score, liquidity/risk, correlation heatmap/cluster,
   M/K/qubit budget và lý do chọn/loại tài sản.
4. Solver Comparison: exact/SA/penalty-QAOA/XY-QAOA; objective, feasibility rate,
   optimality gap, runtime, depth/gates/shots và distribution nếu có.
5. Portfolio & Backtest: weights, trades, costs, equity curve, drawdown, rolling metrics,
   benchmark comparison.
6. Ablation & Robustness: cấu hình ablation, seed distribution, sensitivity và CI.
7. Reproducibility: experiment manifest, config, log và nút tải artifact/report.
- Không khẳng định ưu thế khi chưa có kiểm định. Hiển thị sample size và uncertainty.
- Biểu đồ phải có tiêu đề, đơn vị, legend và nhãn rõ; xử lý trạng thái empty/error.

CHẠY DEMO END-TO-END
- Chọn phạm vi đủ nhỏ để hoàn tất trong môi trường hiện tại nhưng vẫn đi qua tất cả stage:
  data validation → feature/fold → rank → adaptive reduction → instance → ít nhất một
  classical ground truth/baseline và XY-QAOA simulator → weights → một hoặc vài test
  windows → metrics/report.
- Ghi rõ ticker/date/fold/M/K/p/shots/seeds và nguồn dữ liệu.
- Chạy baselines trên đúng input và budget so sánh hợp lý.
- Nếu exact solver chỉ phù hợp instance nhỏ, giới hạn demo và ghi rõ.
- Capture logs, runtime và peak resource nếu có.
- Khởi động UI, thực hiện smoke test các trang/chức năng và kiểm tra không có lỗi console.

XUẤT KẾT QUẢ
Tạo thư mục ổn định, ví dụ outputs/experiments/<experiment_id>/, gồm:
- manifest.json, resolved_config.yaml, environment.txt;
- data_quality.json/md và leakage_audit.json/md;
- fold_manifest và selected_universe;
- QUBO/instance metadata;
- solver_runs và circuit/resource metrics;
- weights.csv, trades.csv, portfolio_returns.csv;
- metrics_long.csv, comparisons.csv, statistical_tests.csv;
- figures/ ở PNG/SVG độ phân giải phù hợp;
- report.html tự chứa hoặc dễ mở cục bộ;
- RESEARCH_REPORT.md tóm tắt phương pháp, dữ liệu, kết quả, uncertainty, giới hạn và cách
  tái lập;
- DEMO_RUN_REPORT.md chứa command, trạng thái, runtime và checksum/artifact index.
Nếu có hỗ trợ PDF ổn định, có thể xuất thêm report.pdf nhưng HTML/Markdown là bắt buộc.

NỘI DUNG BÁO CÁO
- Tách rõ:
  a) data validation results;
  b) predictive/ranking results;
  c) universe reduction results;
  d) optimization/quantum results;
  e) portfolio/backtest results;
  f) ablation/statistical results.
- Liên kết từng nhóm kết quả với H1–H6 và ghi: supported / not supported /
  inconclusive trong phạm vi demo, kèm lý do.
- Demo ngắn không được diễn giải như bằng chứng nghiên cứu toàn kỳ 2015–2025.
- Báo cả kết quả âm, lỗi hội tụ và run bị loại theo tiêu chí định trước.

TEST VÀ ACCEPTANCE CRITERIA
- Tất cả test cũ và mới pass; nếu có skip phải có lý do.
- Một command dựng/chạy quick demo từ môi trường sạch hoặc từ documented prerequisites.
- Không có bước thủ công ẩn để biến raw data thành report.
- Mọi biểu đồ/số trong report truy ngược được đến artifact và experiment manifest.
- Leakage audit pass cho demo; nếu fail thì dừng gắn nhãn research result.
- XY-QAOA output luôn thỏa cardinality trong feasible-subspace test.
- Weight constraints và accounting transaction cost được test.
- UI load được từ artifact đã có ngay cả khi không chạy lại solver.
- README có quickstart Windows/PowerShell phù hợp workspace hiện tại.

YÊU CẦU HOÀN THÀNH
- Triển khai và chạy thật; không chỉ đưa hướng dẫn.
- Sửa lỗi phát hiện trong phạm vi hệ thống nhưng không xóa thay đổi không liên quan.
- Không cài dịch vụ trả phí hoặc dùng secret chưa được cấp.
- Khi bị chặn bởi nguồn/mạng/phần cứng, hoàn thiện phần còn lại, dùng fixture được gắn
  nhãn cho smoke test và ghi blocker chính xác.
- Kết thúc bằng bản tóm tắt: kiến trúc đã hoàn thiện, file chính, command chạy demo,
  URL local của UI nếu đang chạy, artifact/report cuối, test results, kết quả thực tế và
  giới hạn. Không tuyên bố hoàn tất nếu demo end-to-end chưa chạy thành công.
```

---

## Cách sử dụng đề xuất

1. Chạy Prompt 1 trong workspace gốc và chỉ chuyển sang Prompt 2 khi báo cáo coverage,
   quality và leakage audit đã được tạo.
2. Chạy Prompt 2 trong cùng workspace; giữ cấu hình `quick/demo` nhỏ để xác minh trước.
3. Chạy Prompt 3 sau khi quick experiment của Prompt 2 đã có artifact hợp lệ.
4. Với mỗi prompt, lưu lại câu trả lời cuối và `RUN_REPORT.md`; đó là đầu vào kiểm toán
   cho prompt kế tiếp.


# Ma trận đối chiếu báo cáo - hệ thống

Ngày kiểm toán: 2026-08-06  
Baseline: `d952d05` (`main`)  
Nhánh sửa: `fix/research-validity`

Tài liệu này là cổng kiểm soát trước khi sửa mã. Trạng thái `BLOCK` có nghĩa là hệ thống không được phép gắn nhãn kết quả nghiên cứu cho đến khi điều kiện được thỏa mãn bằng dữ liệu hoặc artifact kiểm toán được.

| Tuyên bố/phương pháp trong báo cáo | Hiện trạng trước sửa | Mức độ | Bằng chứng trước sửa | Tiêu chí hoàn thành |
|---|---|---:|---|---|
| Universe HOSE được tái dựng point-in-time | `run_experiment` không đọc `universe_monthly.parquet`; security master của bộ dữ liệu thực được suy từ danh sách hiện tại và ngày đầu có giá | **BLOCK** | `src/research.py:561-625`; `src/sources.py:772-870` | Mỗi fold chỉ dùng snapshot universe có nguồn, hiệu lực, `available_at`; thiếu dữ liệu phải chặn research run |
| Kiểm soát survivorship bias | Báo cáo thừa nhận còn bias nhưng pipeline vẫn cho phép gắn nhãn research | **BLOCK** | `src/research.py:1028` | Leakage audit kiểm tra nội dung/nguồn lịch sử, không chỉ tên cột; fail-closed |
| Observation time khác availability time | Giá có `available_at` nhưng audit trước sửa dùng quan hệ sai (`available_at <= date`) | **BLOCK** | `src/data_pipeline.py:284-287` | Mọi bảng có observation/effective/publication time và `available_at`; không có thông tin trước khi được công bố |
| Walk-forward ngoài mẫu có purging/embargo | Chia theo tháng nhưng nhãn forward 20 phiên có thể vượt biên train/validation; không purge/embargo | **BLOCK** | `src/research.py:74,128-146,599-603` | Lưu `label_end_time`; purge nhãn chạm kỳ sau; áp dụng embargo; artifact fold manifest ghi số dòng bị loại |
| Validation dùng để chọn mô hình | Tham số `validation` được truyền nhưng không dùng | **BLOCK** | `src/research.py:148-160` | Tuning chỉ dùng train/validation; lưu candidate scores, cấu hình chọn và lý do |
| Không impute feature hoàn toàn thiếu | Fundamental/macro có thể 0% coverage nhưng vẫn nằm trong `FEATURES` | **BLOCK** | `src/research.py:29-34,148-164` | Feature coverage theo fold; loại biến dưới ngưỡng trước fit; lưu danh sách active/dropped |
| XGBoost được so sánh độc lập với EWMA | Có signal EWMA ở một ablation nhưng chưa có Rank IC cặp và kiểm định H1 | **MAJOR** | `src/research.py:722-729` | Lưu IC theo fold cho cả hai; bootstrap/permutation cặp và Holm khi cần |
| Adaptive Universe Reduction có M thích ứng | Thành viên thích ứng nhưng kích thước luôn `min(candidate_size, qubit_budget)` | **MAJOR** | `src/research.py:167-197` | M thay đổi theo dispersion/tương quan/thanh khoản trong biên cấu hình; lưu diagnostics và lý do |
| AUR xét tín hiệu, thanh khoản, rủi ro, tương quan, qubit | Có điểm tổng hợp nhưng diagnostics không đủ để tái lập | **MAJOR** | `src/research.py:167-197,620-624` | Lưu từng thành phần score, correlation penalty, eligible count, selected M, turnover universe |
| QUBO và ánh xạ Ising minh bạch | Chỉ lưu ma trận QUBO | **MAJOR** | `src/research.py:199-201,643-649` | Lưu Q, h, J, offset; test đồng nhất năng lượng QUBO-Ising |
| XY-QAOA/Dicke tối ưu tham số biến phân | Trước sửa chỉ random-search góc | **BLOCK** | `src/research.py:305-358` | Dùng optimizer cổ điển có trace, multi-start seed, stopping reason, params tốt nhất |
| Penalty-QAOA có ngân sách tối ưu công bằng | Có statevector circuit nhưng cũng random-search; ngân sách không đối xứng trong ablation | **BLOCK** | `src/research.py:375-437,753-764` | Cùng optimizer family/budget/seeds/shots/depth; lưu penalty strength và traces |
| Nghiệm QAOA chính là nghiệm có xác suất cao nhất | Trước sửa chọn nghiệm có energy thấp nhất trong các mẫu đã thấy | **BLOCK** | `src/research.py:347-354,422-431` | Primary = most-probable feasible bitstring; best-observed báo riêng; lưu expected energy và success probability |
| So sánh optimality gap có ý nghĩa | Gap thường bằng 0 do cách chọn best-observed | **BLOCK** | `src/research.py:664-671` | Gap chính dựa trên primary solution; exact chỉ làm oracle ở instance nhỏ; báo best-observed phụ |
| Noise được mô tả đúng bản chất | Trộn xác suất đều được gọi chung là noise | **MAJOR** | `src/research.py:306,344-346` | Đổi tên thành uniform-probability noise proxy hoặc triển khai noise model thực; không gọi là hardware noise |
| Full pipeline thực sự dùng XY-QAOA | Strategy chính tên `full_pipeline_exact_selection`; một ablation khác mới dùng XY | **BLOCK** | `src/research.py:672-706,708-716` | Strategy chính chọn bằng XY-QAOA; exact/SA/penalty là comparator tách biệt |
| Benchmark độc lập và công bằng | Thiếu Markowitz, minimum variance, universe 1/N; baseline phần lớn không chịu phí | **BLOCK** | `src/research.py:680-716,784-788` | Benchmark độc lập: universe 1/N, candidate 1/N, Markowitz, min-var, liquidity, EWMA, XGB, exact, SA, penalty, XY; cùng lịch và phí |
| Buy-and-hold giữa hai lần tái cân bằng | Return được tính bằng fixed weights mỗi ngày nhưng cuối kỳ lại drift weights | **BLOCK** | `src/research.py:694-700,777-788` | Mô phỏng số đơn vị/giá trị tài sản, weight drift hàng ngày; rebalance đúng kỳ |
| Turnover và chi phí nhất quán mọi chiến lược | Chỉ cấu hình `08_full_pipeline_costs` chịu phí; equal weight không có ledger | **BLOCK** | `src/research.py:700-716,784-787` | Mọi chiến lược có target/pre-trade weights, trades, turnover, gross/net return và cost ledger |
| Tối ưu tỷ trọng có ràng buộc thực tế | Có long-only, tổng bằng 1, upper bound và turnover penalty; chi phí thực hiện hậu kiểm | **PARTIAL** | `src/research.py:477-525` | Kiểm tra khả thi lower/upper; công khai turnover penalty và cost accounting; test constraints |
| Sensitivity H6 là thực nghiệm thật | Cost labels được nhân bản mà không chạy lại return; chỉ fold 0 | **BLOCK** | `src/research.py:805-827` | OFAT thực sự chạy lại nghiệm/backtest/cost; khai báo grid, seeds và folds đại diện |
| Ablation đo đóng góp từng khối | Có 8 cấu hình nhưng không hoàn toàn cùng chi phí và solver budget | **MAJOR** | `src/research.py:707-804` | Thiết kế ablation một-yếu-tố, cùng dữ liệu/lịch/chi phí; artifact mô tả cấu hình |
| Block bootstrap kiểm định chênh lệch | Bootstrap trước sửa không centered theo null | **MAJOR** | `src/research.py:453-475` | Bootstrap cặp theo block, centered under null; CI, p-value, Holm, effect size |
| H1-H6 được đánh giá đúng tầng | Báo cáo tự động chủ yếu mô tả, H2 chưa kiểm định | **MAJOR** | `src/research.py:957-1041` | Bảng H1-H6 liên kết artifact, tiêu chí, kết quả và giới hạn suy luận |
| Coverage 2015-2025 và số folds được công bố đúng | Config thực hiện hiện tại là 2020-2025, max 12 fold, OOS chủ yếu 2025 | **BLOCK** | `configs/hose300_real.yaml` | Manifest ghi requested/actual data range, OOS range, folds; không tuyên bố 2015-2025 nếu dữ liệu không có |
| Corporate actions/adjusted price được kiểm soát | Bảng sự kiện thực chưa được xác minh đầy đủ | **BLOCK** | `src/data_pipeline.py:256-308` | Research run yêu cầu provenance và action/adjustment policy được kiểm toán, hoặc chặn |
| Artifacts đủ tái lập và kiểm toán | Có nhiều CSV/JSON nhưng thiếu optimizer traces, feature coverage, universe diagnostics, cost ledger, environment lock | **MAJOR** | `src/research.py:828-949` | Bổ sung manifest hash, provenance, folds, coverage, tuning, QAOA traces, diagnostics, cost ledger, audit script |
| Môi trường tái lập | Dependencies phần lớn không pin phiên bản | **MAJOR** | `pyproject.toml` | Pin direct dependencies và sinh `requirements.lock`; lưu Python/platform/package versions |
| Encoding UTF-8 sạch | README/config/report có mojibake | **MAJOR** | `configs/hose300_real.yaml`; `src/research.py`; `RUN_REPORT.md` | Quét encoding, sửa văn bản nguồn, test không có marker mojibake |
| Kết quả GitHub nhẹ và trung thực | Outputs bị ignore; repo không có gói latest research có provenance | **MAJOR** | `.gitignore` | `results/latest_research/` chỉ chứa artifacts nhẹ hoặc blocker manifest; không commit raw data/secrets |
| Kiểm thử bao phủ research validity | Baseline chỉ 16 tests | **MAJOR** | `tests/` | Test PIT, purging, fold, QUBO-Ising, QAOA primary, buy-and-hold, costs, AUR, bootstrap, artifacts, encoding, determinism |

## Kết luận cổng kiểm soát trước sửa

Hệ thống trước sửa chạy được về mặt phần mềm, nhưng chưa đủ điều kiện để gọi kết quả hiện có là bằng chứng nghiên cứu xác nhận. Blocker dữ liệu lớn nhất là chưa có universe HOSE lịch sử có hiệu lực và thời điểm công bố được xác minh. Việc sửa mã sẽ không tự chế tạo nguồn này; nếu artifact đầu vào vẫn thiếu, research mode phải tạo báo cáo `blocked` rõ nguyên nhân và dừng trước huấn luyện/backtest.

## Trạng thái sau sửa

| Nhóm gap | Trạng thái | Bằng chứng sau sửa |
|---|---|---|
| Historical universe, survivorship, corporate actions, adjustment policy | **BLOCKED — external verified data required** | Research gate dừng trước model; `results/latest_research/blocker_manifest.json` |
| Observation/availability semantics và provenance | **RESOLVED IN CODE** | `data_pipeline.py`, `sources.py`, `DATA_GOVERNANCE_AND_PIT.md` |
| Purging/embargo, fold boundaries, feature coverage, validation tuning | **RESOLVED** | `fold_manifest.csv`, `feature_coverage_by_fold.csv`, `model_tuning.csv`; automated tests |
| XGBoost–EWMA H1 | **RESOLVED FOR EXECUTION; RESEARCH RESULT BLOCKED** | Rank IC theo fold và centered paired block bootstrap |
| Adaptive M và diagnostics | **RESOLVED** | `aur_diagnostics.csv`, `selected_universe.csv` |
| QUBO–Ising và QAOA optimizer | **RESOLVED** | Ising artifacts; COBYLA multi-start traces; energy-equivalence tests |
| Primary vs best-observed QAOA solution | **RESOLVED** | `solver_runs.csv` có primary probability, expected energy, best-observed và hai gap |
| Full pipeline dùng XY-QAOA | **RESOLVED** | Strategy `full_pipeline_xy_qaoa`; exact là oracle riêng |
| Benchmark, buy-and-hold và common transaction costs | **RESOLVED** | universe/candidate 1/N, Markowitz, min-var, selector/solver comparators; `cost_ledger.csv` |
| Bootstrap/Holm, ablation, sensitivity | **RESOLVED FOR SOFTWARE; RESEARCH INFERENCE BLOCKED** | centered paired block bootstrap; real reruns theo grid/folds/seeds |
| Artifacts, audit, UTF-8, dependency lock | **RESOLVED** | `audit_research_run.py`, 36 tests, `requirements.lock`, documentation set |

Kết luận sau sửa: demo fixture chạy và audit `pass`; panel giá thật 300 mã vượt data-quality
gate nhưng research run audit là `blocked_valid`. Đây là trạng thái đúng theo yêu cầu
fail-closed, không phải lỗi thực thi và không được chuyển thành kết quả nghiên cứu bằng cách
hạ tiêu chuẩn kiểm toán.

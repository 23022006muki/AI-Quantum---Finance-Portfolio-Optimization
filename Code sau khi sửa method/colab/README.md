# Hướng dẫn chạy Colab standalone

## File chính

- `AUR_QAUR_XYQAOA_Standalone_Full_Colab.ipynb`: toàn bộ code được viết trực
  tiếp trong notebook; không clone hoặc tải source từ GitHub.

## Dữ liệu upload

Sử dụng một trong hai file đã có trong repository:

- `quantum_portfolio_data/colab_data/ai_quantum_complete_dataset.zip`;
- `quantum_portfolio_data/colab_data/ai_quantum_complete_dataset.csv`.

ZIP nhỏ hơn và được khuyến nghị khi upload lên Colab.

## Cách chạy

1. Upload notebook lên Google Colab.
2. Chọn Runtime -> Change runtime type -> CPU (GPU không bắt buộc).
3. Để `EXECUTION_PROFILE = "SMOKE"`, chạy tất cả cell một lần để kiểm tra.
4. Nếu smoke run hoàn tất, đổi thành `EXECUTION_PROFILE = "FULL"`.
5. Chọn Runtime -> Restart session, sau đó Run all.
6. Khi được hỏi, upload đúng một file CSV hoặc ZIP nêu trên.
7. Cell cuối tự tải `AUR_QAUR_XYQAOA_RESULTS.zip`.

## Lưu ý phương pháp

QAUR sử dụng classical cardinality-preserving surrogate cho quantum-ready
universe-reduction QUBO. Shared XY-QAOA là ideal statevector simulation trong
fixed-Hamming-weight feasible subspace. Notebook không tuyên bố quantum advantage.

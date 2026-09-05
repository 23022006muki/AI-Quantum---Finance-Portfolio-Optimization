# Data sau khi sửa method

Bộ dữ liệu này được version hóa cho framework so sánh **Adaptive Universe
Reduction (AUR)** với **Quantum-Assisted Universe Reduction (QAUR)**, trong đó
Cardinality-Constrained QUBO, XY-QAOA, tối ưu tỷ trọng và walk-forward backtest
được dùng chung ở downstream.

File `data_sau_khi_sua_method.zip` là file khuyến nghị để upload vào notebook
Colab. ZIP chứa đúng một CSV. Các quan sát thị trường không bị thay đổi so với
dataset nguồn; chỉ metadata dataset được cập nhật để phản ánh phương pháp mới.

`manifest.json` lưu hash, số dòng, khoảng thời gian, record counts và các kiểm
tra chất lượng. `schema.json` lưu toàn bộ cột và kiểu dữ liệu.

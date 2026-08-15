# External data access required

Pipeline code có thể chạy và kiểm toán, nhưng research run thật chưa thể hoàn thành nếu thiếu
quyền truy cập vào các nguồn dữ liệu dưới đây. Không gửi secret qua Git hoặc dán trực tiếp vào code.

## 1. Vietstock

Phiên/cookie đã từng được chia sẻ trước đây không được tái sử dụng. Hãy đăng xuất rồi đăng nhập lại,
sau đó lưu cookie của một request lịch sử thành file văn bản nằm ngoài repository và đặt:

```powershell
$env:VIETSTOCK_COOKIE_FILE="C:\path\outside-repository\vietstock-cookie.txt"
```

Hoặc lưu request headers dạng JSON vào file ngoài repository rồi đặt
`VIETSTOCK_AUTH_HEADER_FILE`. Không đưa hai file này lên Git.

## 2. SSI FastConnect

Tạo key được cấp quyền tại SSI và cấu hình trong phiên PowerShell:

```powershell
$env:SSI_CONSUMER_ID="..."
$env:SSI_CONSUMER_SECRET="..."
```

## 3. Trading Economics

Cần API key có quyền truy cập historical markets:

```powershell
$env:TRADING_ECONOMICS_API_KEY="..."
```

Trading Economics chỉ được dùng để đối chiếu OHLC. API lịch sử không cung cấp đủ volume,
corporate-action adjustment hoặc lịch sử listing để thay security master HOSE.

## 4. Các lớp point-in-time còn bắt buộc

- Security master HOSE 2015–2025 đã được lấy từ nguồn chính thức bằng
  `crawl-hose-security-master`; không cần cung cấp lại trừ khi muốn mở rộng giai đoạn.
- Panel OHLC hiện bao phủ 444/445 mã giao cắt giai đoạn 2020–2025. Mã VPK niêm yết đến
  13/01/2020 nhưng không có phiên HOSE quan sát được trong giai đoạn trên các nguồn công khai
  đã thử; cần dữ liệu exchange/vendor có cấp phép hoặc quy tắc ngoại lệ no-trading được thẩm định.
- Corporate actions theo `docs/contracts/corporate_actions_template.csv`.
- Total-return benchmark theo `docs/contracts/benchmark_template.csv`.
- Tài liệu phương pháp điều chỉnh giá để hoàn thiện price-adjustment contract.

Sau khi cấu hình, chạy:

```powershell
python -m src.cli prepare-research-data --config configs/hose300_real.yaml
```

Lệnh chỉ in boolean credential và trạng thái contract, không in giá trị secret.

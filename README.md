# MT-ANH-VIET: Dịch máy Anh–Việt – Seq2seq Attention vs Transformer

Bài tập lớn môn Học sâu – Đề tài 8.

## Thành viên
| Họ tên | MSSV | Phụ trách |
|---|---|---|
| Văn Đức Hiếu | 24022331 | Dữ liệu, tokenizer, đánh giá |
| Lương Quang Hoàng | 24022337 | Seq2seq LSTM + attention |
| Ngô Trọng Hiệp | 24022325 | Transformer |

## Bài toán
Đầu vào: câu tiếng Anh. Đầu ra: câu dịch tiếng Việt + attention map.
Dữ liệu: IWSLT'15 En–Vi (dev: tst2012, test: tst2013). Đánh giá: SacreBLEU.

## Cài đặt
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# cài torch theo hướng dẫn tại pytorch.org (chọn đúng CUDA), sau đó:
pip install -r requirements.txt
```

## Dữ liệu
Phần dữ liệu/tokenizer/đánh giá do Văn Đức Hiếu phụ trách đã được triển khai.
Chạy từ thư mục gốc:

```bash
python scripts/download_data.py
# Nếu Stanford timeout: python scripts/download_data.py --source mirror
python scripts/prepare_data.py
python scripts/train_bpe.py
python -m pytest -q
```

Dữ liệu không nằm trong repo. Download xác minh SHA-256 của cả 6 file.
Sau xử lý: **132.137 cặp train**, **1.553 dev**, **1.268 test**.
BPE chung học **8.000 phép gộp chỉ từ train**; vocabulary gồm 8.360 token.
Reference dev/test giữ nguyên byte và thứ tự.

Xem [hướng dẫn tái lập và bàn giao cho Hoàng/Hiệp](docs/data-and-evaluation.md),
[kiểm tra căn chỉnh](reports/alignment_audit.md),
[thống kê dữ liệu](reports/data_statistics.json) và [thống kê BPE](reports/bpe_statistics.json).

## Huấn luyện
Phần LSTM do Lương Quang Hoàng, Transformer do Ngô Trọng Hiệp triển khai tiếp.
Chưa có `src.train` hoặc checkpoint đã huấn luyện.
Hai mô hình dùng `src.dataset.TranslationDataset`, `collate_batch` và
`src.tokenizer.SharedTokenizer` theo giao diện trong tài liệu bàn giao.

## Đánh giá
`python -m src.evaluate score` chấm file dự đoán bằng SacreBLEU 13a, phân biệt hoa/thường.
`python -m src.evaluate freeze` chọn checkpoint/cấu hình chỉ từ kết quả **dev**.
Chấm **test** bắt buộc có selection đã khóa và các hash trùng khớp.
Xem lệnh đầy đủ trong [tài liệu](docs/data-and-evaluation.md#chấm-điểm-và-khóa-test).

## Kết quả
Chưa có kết quả mô hình thật. Khi có predictions, chạy
`python scripts/summarize_results.py --results <file-json...> --output-dir results`
để tạo bảng BLEU, số tham số, thời gian, tốc độ giải mã và mean/std theo seed.
Xem [trạng thái kết quả](results/README.md).

## Cấu trúc thư mục
- `scripts/`: download, tiền xử lý, học BPE, tổng hợp bảng.
- `src/`: dữ liệu, tokenizer, batch, scorer; `src/models/` dành cho hai kiến trúc.
- `configs/`: nguồn dữ liệu/checksum và danh sách loại có lý do.
- `reports/`: thống kê thực tế và kiểm tra căn chỉnh.
- `docs/`: giao diện tích hợp và quy trình không dùng test để tuning.
- `tests/`: kiểm thử bằng fixture nhỏ; CI không cần tải corpus.

## Công cụ AI đã dùng
Codex hỗ trợ triển khai và kiểm thử pipeline dữ liệu, BPE, batch, SacreBLEU,
quy trình dev/test, thống kê và tài liệu bàn giao; hỗ trợ rà soát mẫu căn chỉnh.
Thành viên cần đọc, hiểu và kiểm tra lại trước khi nộp; chưa có kết quả huấn luyện
nào được tạo hoặc giả lập để báo cáo như thực nghiệm thật.

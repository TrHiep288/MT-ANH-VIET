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
TODO: `python scripts/download_data.py` (dữ liệu không nằm trong repo).

## Huấn luyện
TODO: `python -m src.train --config configs/seq2seq.yaml`
TODO: `python -m src.train --config configs/transformer.yaml`

## Đánh giá
TODO: `python -m src.evaluate --ckpt <path>`

## Kết quả
TODO (bảng BLEU, số tham số, thời gian huấn luyện, tốc độ giải mã).

## Cấu trúc thư mục
Xem cây thư mục trong repo. Log huấn luyện: Weights & Biases (link: TODO).

## Công cụ AI đã dùng
TODO (khai báo đã dùng vào việc gì, theo yêu cầu của đề).
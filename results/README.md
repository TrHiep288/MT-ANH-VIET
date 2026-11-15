# Kết quả thực nghiệm

Chưa có checkpoint hoặc predictions từ LSTM/Transformer, nên chưa có BLEU hay
bảng so sánh mô hình thật. Không dùng kết quả fixture unit test làm kết quả báo cáo.

Sau huấn luyện, làm theo [quy trình dev → freeze → test](../docs/data-and-evaluation.md).
`scripts/summarize_results.py` tạo `runs.csv` và `summary.md` từ file kết quả thật.
Hai mô hình cần cùng dữ liệu, BPE, thiết bị và giao thức đo. Báo cáo số seed thực tế;
không điền ba seed khi chỉ chạy một seed.

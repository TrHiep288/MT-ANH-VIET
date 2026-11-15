# Bàn giao dữ liệu và đánh giá — Văn Đức Hiếu (24022331)

## Phạm vi và giao diện với hai thành viên

Hoàng triển khai LSTM hai chiều + decoder LSTM, Bahdanau attention, teacher forcing,
greedy decoding và huấn luyện. Hiệp triển khai Transformer, positional encoding,
attention/mask, Adam/warm-up và huấn luyện. Nhánh này cung cấp dữ liệu, tokenizer,
batch và chấm điểm dùng chung; chưa có mô hình hoặc checkpoint đã huấn luyện.

Hai mô hình dùng **cùng** `data/manifest.json`, `data/bpe/metadata.json`, vocabulary
và reference. Khi lưu checkpoint, lưu thêm hash của hai manifest này. Không tự
train lại tokenizer riêng, chia lại tập, lowercase, bỏ dấu, giải mã HTML entity,
hoặc lọc dev/test. Quy tắc chuẩn hóa là NFC và gộp khoảng trắng.

## Tái lập dữ liệu

Chạy từ thư mục gốc, Python 3.11:

```bash
python -m pip install -r requirements-data.txt
python scripts/download_data.py
# Nếu máy chủ Stanford không phản hồi, dùng mirror cố định commit:
python scripts/download_data.py --source mirror
python scripts/prepare_data.py
python scripts/train_bpe.py
python -m pytest -q
```

Cài thêm PyTorch để dùng `src.dataset` và kiểm thử batch. Data/evaluation không
yêu cầu GPU. `requirements.txt` giữ các thư viện phục vụ phần mô hình của nhóm.

Nguồn chính: [Stanford NMT](https://nlp.stanford.edu/projects/nmt/).
Mirror: [stefan-it/nmt-en-vi, commit cố định](https://github.com/stefan-it/nmt-en-vi/tree/93193ddf269449c98495e3455b2bb261b4851234/data).
SHA-256 trong `configs/data_sources.json` đối chiếu với
[bản ghi checksum của dataset IWSLT](https://huggingface.co/datasets/IWSLT/mt_eng_vietnamese/commit/bdd7c9514c01e32ac452a9ae25aac953cc944555).
Chỉ sử dụng checksum từ bản ghi đó; **không dùng nhãn split trong wrapper**.
Ánh xạ được khóa bằng tên file: train → train, dev → tst2012, test → tst2013.
Mirror chỉ được chấp nhận nếu nội dung khớp byte với checksum nguồn chính.
Không tải hoặc chạy code từ mirror. Nguồn thực tế ghi ở `data/download_receipt.json`.

| Split | Gốc | Sau xử lý |
|---|---:|---:|
| train | 133317 | 132137 |
| dev / tst2012 | 1553 | 1553 |
| test / tst2013 | 1268 | 1268 |

Loại 151 dòng có ít nhất một vế rỗng, 621 **lần xuất hiện** của cặp trùng với dev/test,
406 cặp train trùng còn lại và 2 dòng sai cặp đã kiểm tra thủ công.
Thứ tự ưu tiên: loại thủ công → rỗng → trùng held-out → trùng train;
mỗi dòng bị loại chỉ được tính một lần. Số overlap trong báo cáo là số **cặp duy nhất**,
không phải số lần xuất hiện, nên khác 621. Không lọc theo độ dài hoặc tỉ lệ độ dài.

Hai dòng train gốc 133316–133317 nói về hai bài TED khác nhau (Didier Sornette vs
Paul Pholeros), được loại theo `configs/train_exclusions.json`. ID này chỉ áp dụng
sau khi xác minh toàn bộ checksum của corpus. Xem `reports/alignment_audit.md`.

Vẫn có overlap một vế (cùng câu nguồn nhưng bản dịch khác, hoặc ngược lại),
và 3 cặp duy nhất chung giữa dev/test; báo cáo nêu rõ. Giữ nguyên dev/test theo proposal.
Không tuyên bố dữ liệu hoàn toàn sạch hoặc mọi cặp đều đã được kiểm tra ngữ nghĩa.

Các file được tạo, không commit corpus/checkpoint vào Git:

| File | Ý nghĩa |
|---|---|
| `data/raw/{train,tst2012,tst2013}.{en,vi}` | Nội dung nguồn nguyên bản |
| `data/processed/{train,dev,test}.{en,vi}` | Cặp đã chuẩn hóa, cùng thứ tự |
| `data/processed/{train,dev,test}.ids` | Số dòng gốc, bắt đầu từ 1 |
| `data/references/{dev,test}.vi` | Bản sao nguyên byte của reference gốc, không BPE |
| `data/manifest.json` | Hash từng file và chính sách xử lý |
| `data/bpe/codes.bpe`, `vocab.json` | BPE và vocabulary chung |
| `data/bpe/{train,dev,test}.{en,vi}` | Chuỗi subword, marker `@@` |
| `data/bpe/metadata.json` | Hash, số merge, phiên bản, token đặc biệt, OOV |
| `reports/*_statistics.json` | Thống kê thực tế có thể tái lập |

`subword-nmt==0.3.8` học **8.000 phép gộp**, không phải vocab size 8.000,
từ train.en + train.vi đã làm sạch. Vocab được đếm từ subword train, xếp theo
tần suất giảm dần rồi thứ tự Unicode để ổn định. Dev/test chỉ được apply BPE;
token không có trong vocab train trở thành `<unk>`. Không thêm ký tự từ test vào vocab.
[Tài liệu subword-nmt](https://github.com/rsennrich/subword-nmt).

## Batch dùng chung cho LSTM và Transformer

```python
from torch.utils.data import DataLoader
from src.dataset import TranslationDataset, collate_batch
from src.tokenizer import PAD

train = TranslationDataset("data", "train")
dev = TranslationDataset("data", "dev")
loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate_batch)
dev_loader = DataLoader(dev, batch_size=32, shuffle=False, collate_fn=collate_batch)
tokenizer = train.tokenizer
vocab_size = len(tokenizer)  # không hard-code 8000
batch = next(iter(loader))
# logits = model(batch["src"], batch["tgt_in"], ...)
# loss = torch.nn.functional.cross_entropy(
#     logits.reshape(-1, vocab_size), batch["tgt_out"].reshape(-1), ignore_index=PAD)
```

| Thành phần | Quy ước |
|---|---|
| IDs | PAD=0, UNK=1, BOS=2, EOS=3; source và target dùng cùng vocab |
| `src` | `LongTensor[B,S]`, source + EOS, right-padding |
| `src_lengths` | `LongTensor[B]`, gồm EOS, chưa sort; cho pack dùng `enforce_sorted=False` |
| `tgt_in` | `LongTensor[B,T]`, BOS + target subwords |
| `tgt_out` | `LongTensor[B,T]`, target subwords + EOS |
| `src_padding_mask` | `BoolTensor[B,S]`, True = vị trí phải bỏ qua |
| `tgt_padding_mask` | `BoolTensor[B,T]`, mask của decoder input |
| `tgt_causal_mask` | `BoolTensor[T,T]`, True phía trên đường chéo = không được nhìn |
| `ids` | ID dòng gốc, dùng nối dự đoán và mẫu phân tích |

Hoàng dùng source mask khi tính Bahdanau attention; Hiệp kết hợp causal mask
và target padding mask, đồng thời dùng source mask cho encoder/cross-attention.
Hai người phải thống nhất chiều batch-first; nếu module dùng time-first thì transpose.
Batch nằm trên CPU; vòng train chuyển tensor cần dùng sang device.
Không tự thêm BOS/EOS lần nữa. Không truncate câu một cách âm thầm.
Đặc biệt train có câu dài (xem thống kê); nếu đổi chính sách độ dài, cần thống nhất
giữa hai mô hình và ghi lại cấu hình, chỉ chọn bằng dev.

Inference bắt đầu bằng BOS, kết thúc tại EOS hoặc giới hạn độ dài đã chọn trên dev.
`tokenizer.decode(ids)` bỏ BOS/PAD, dừng ở EOS, ghép BPE. Ghi đúng một dòng mỗi câu,
giữ cả dòng rỗng khi mô hình không sinh token. Với dev/test dùng `shuffle=False`,
không bỏ câu, không đổi thứ tự (hoặc sắp lại theo `ids` trước khi ghi).
Model không được dùng `tgt_in`/reference để sinh bản dịch đánh giá.

## Chấm điểm và khóa test

Scorer nhận **file dự đoán**, không tự load kiến trúc hoặc chạy checkpoint.
Hai thành viên cung cấp checkpoint thật, cấu hình JSON đã dùng, và predictions
của chính checkpoint đó. Config ghi đủ kiến trúc, optimizer, learning rate,
dropout, preprocessing, giới hạn decode và beam/length penalty nếu có.
Seed truyền riêng bằng `--seed`; để tổng hợp nhiều seed, dùng cùng config JSON
và không nhúng seed/đường dẫn checkpoint vào file cấu hình chung.

Ví dụ sau khi Hoàng đã tạo checkpoint và predictions:

```bash
python -m src.evaluate score --split dev --model seq2seq --seed 42 --checkpoint checkpoints/seq2seq-42.pt --config configs/seq2seq-run.json --hyp results/seq2seq-42.dev.txt --output results/runs/seq2seq-42.dev.json
python -m src.evaluate freeze --dev-results results/runs/seq2seq-42.dev.json --output results/selections/seq2seq-42.json
python -m src.evaluate score --split test --model seq2seq --seed 42 --checkpoint checkpoints/seq2seq-42.pt --config configs/seq2seq-run.json --hyp results/seq2seq-42.test.txt --selection results/selections/seq2seq-42.json --output results/runs/seq2seq-42.test.json
```

Đổi `seq2seq` thành `transformer` và dùng checkpoint/config/predictions tương ứng
cho Hiệp. Khi thử nhiều checkpoint/cấu hình trên dev, truyền **toàn bộ** file dev
cùng model và seed vào `--dev-results`; freeze chọn BLEU lớn nhất, hòa chọn file
đầu tiên. Dùng file kết quả mới cho từng epoch/trial. Early stopping (patience 4)
và tối đa 30 epoch do vòng train của từng thành viên thực hiện, không do scorer.

Scorer dùng `sacrebleu==2.5.1`, `tokenize=13a`, `lowercase=False`, corpus BLEU,
exp smoothing và effective_order=False. Lưu đầy đủ signature, hash reference,
checkpoint, config, tokenizer, dữ liệu và predictions.
Reference Stanford đã được token hóa sẵn và có HTML entities; giữ nguyên theo
proposal, không tự detokenize hoặc unescape. Predictions phải cùng quy ước văn bản
với target huấn luyện. Chỉ ghép subword, **không biến đổi reference**.
Nếu ghi chuỗi BPE thay vì `decode`, thêm `--hyp-format bpe`.
[Tài liệu SacreBLEU](https://github.com/mjpost/sacrebleu).

Test bị từ chối nếu thiếu selection từ dev hoặc hash checkpoint/config/tokenizer/
dữ liệu, model, seed hay decoding khác selection. Ghi file dùng chế độ không đè.
Đây là **cơ chế kiểm tra quy trình**, không phải kiểm soát truy cập: người có dữ
liệu vẫn có thể xem test, sửa JSON hoặc tạo selection mới. Script không thể chứng
minh file predictions thực sự do checkpoint khai báo sinh ra. Nhóm cần review
training loop: chỉ train để cập nhật trọng số, chỉ dev để chọn epoch/siêu tham số;
chốt toàn bộ trial trước khi test, không chọn lại dựa trên BLEU test.

Beam4 là mở rộng: chọn cấu hình trên dev, ghi đầy đủ giới hạn/penalty vào config,
freeze với `--decoding beam4`, báo riêng khỏi greedy. Không so trực tiếp BLEU này
với con số 23,3 từ tài liệu nếu quy trình chấm khác.

## Tổng hợp kết quả thật

Truyền thêm `--parameters`, `--train-seconds`, `--decode-seconds`, `--device`
ở lệnh score để ghi chi phí thực nghiệm. `decode-seconds` là thời gian sinh toàn
bộ split, không gồm scorer; với CUDA đồng bộ trước và sau đo. So sánh hai mô hình
trên cùng thiết bị, batch size và cách đo. Giá trị chưa đo để trống, không dùng 0 giả.

```bash
python scripts/summarize_results.py --results results/runs/seq2seq-42.test.json results/runs/transformer-42.test.json --output-dir results
```

Tạo `runs.csv` (BLEU, số tham số, thời gian, câu/giây, signature và hash) và
`summary.md` (mean và sample std theo seed). Chỉ gộp cấu hình, device, dữ liệu,
tokenizer, reference và metric giống nhau. Một seed có std=N/A. Mỗi nhóm/seed
chỉ nhận một kết quả; không tự chọn bản tốt nhất bằng test. Hiện chưa có kết quả
mô hình thật vì hai phần model/training chưa được triển khai.

Sau khi khóa lựa chọn và hoàn tất test, nhóm thực hiện phân tích 20 câu theo độ
dài nguồn và attention maps theo proposal; đây là phần phân tích chung, không
phải kết quả được tạo sẵn trong nhánh này.

## Kiểm thử

`python -m pytest -q` kiểm tra chuẩn hóa tiếng Việt, sai số dòng/checksum, lọc cặp
và ID, bảo toàn reference, BPE chỉ học train, roundtrip, batch/mask/target shift,
BLEU, khóa test, từ chối đổi checkpoint và tổng hợp kết quả. Fixture BLEU dùng
dữ liệu nhỏ tổng hợp, không được báo như chất lượng của mô hình thật.
CI chạy kiểm thử CPU; không tải corpus trong mỗi CI run.

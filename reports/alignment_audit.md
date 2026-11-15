# Kiểm tra căn chỉnh và đúng split

Người phụ trách: Văn Đức Hiếu. Kiểm tra bằng script và rà soát mẫu có hỗ trợ Codex.

Tải từ mirror commit `93193ddf269449c98495e3455b2bb261b4851234` vì Stanford timeout.
Cả 6 file khớp SHA-256 nguồn gốc đã khóa trong `configs/data_sources.json`.
Số dòng mỗi vế: train=133317, tst2012=1553, tst2013=1268.
Không shuffle hai vế riêng; giữ số dòng gốc trong file `.ids`.

Rà soát ngữ nghĩa mẫu theo ID dòng **raw**, không dùng để chọn siêu tham số:

| File | Dòng | Nhận xét |
|---|---:|---|
| train | 1 | Cùng tiêu đề khoa học/khí hậu; bản Việt thiếu tên diễn giả, giữ nguyên |
| train | 66659 | Cùng ý phụ nữ có khả năng làm việc này hơn đàn ông |
| train | 133306–133314 | Cùng nội dung môi trường sống, nghèo đói, kết luận và cảm ơn |
| train | 133315 | Cả hai vế rỗng; bộ lọc rỗng loại |
| train | 133316 | Sai cặp tiêu đề: Sornette/khủng hoảng tài chính vs Pholeros/nhà ở |
| train | 133317 | Sai cặp mô tả: dự đoán khủng hoảng vs Healthabitat/cải thiện nhà ở |
| tst2012 | 1, 777, 1553 | Cùng ý ở từng cặp: gắn kết phụ nữ; bận/vội; câu hỏi 50 năm tới |
| tst2013 | 1, 635, 1268 | Cùng ý ở từng cặp: quê hương/bài hát; câu hỏi cho bé gái; cảm ơn |

Loại **chỉ hai dòng train 133316–133317** theo danh sách lý do được version-control.
Các hash nguồn được kiểm tra trước khi áp dụng danh sách loại.
Không suy diễn rằng toàn bộ 133317 cặp đã được con người xác nhận: kiểm tra tự
động xác nhận cấu trúc/nguồn/thứ tự; rà soát ngữ nghĩa chỉ bao phủ các mẫu trên.

`data_statistics.json` ghi số cặp rỗng/trùng/overlap, số câu, độ dài và chính sách.
Sau xử lý còn 132137 cặp train, dev/test giữ nguyên. Có 3 cặp duy nhất chung
giữa dev và test, cùng các overlap một vế; giữ nguyên theo proposal và báo cáo
giới hạn này khi trình bày kết quả. Không dùng test để chọn cấu hình BPE/mô hình.

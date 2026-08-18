# Failure Analysis — Lab 18: Production RAG

**Hình thức:** Cá nhân · **Người thực hiện:** Bùi Đặng Quốc An · **MSSV:** 2A202601799

**Lưu ý đánh giá:** Báo cáo JSON hiện tại được tạo ở lần chạy offline trước khi cấu hình API key, vì vậy các số dưới đây là lexical proxy được ghi rõ trong JSON, không phải điểm RAGAS do LLM chấm. Chạy lại `python main.py` với quyền gửi dữ liệu ra OpenAI để tạo điểm RAGAS thật.

## Kết quả đánh giá

| Metric | Naive baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer relevancy | 0.7587 | 0.8044 | +0.0458 |
| Context precision | 1.0000 | 1.0000 | +0.0000 |
| Context recall | 0.7422 | 0.8785 | +0.1363 |

Kết quả đáng chú ý nhất là context recall tăng 0.1363. Parent-child chunking và hybrid retrieval tìm được nhiều bằng chứng hơn baseline dense-only. Precision proxy không đổi vì phép đo lexical chỉ kiểm tra context có giao từ khóa với câu hỏi/ground truth; cần RAGAS thật để đánh giá ngữ nghĩa và độ nhiễu chính xác hơn.

## Diagnostic Tree

```text
Output sai hoặc thiếu?
├─ Context thiếu bằng chứng? → Context recall → chunking, top-k, query expansion
├─ Context có quá nhiều nhiễu? → Context precision → reranker, metadata/version filter
├─ Context đúng nhưng câu trả lời lạc đề? → Answer relevancy → prompt/extractive selector
└─ Câu trả lời có claim ngoài context? → Faithfulness → grounded prompt, citation
```

## Bottom-5 Failures

### 1. Mua laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt; CNTT xác nhận cấu hình; đính kèm ít nhất 3 báo giá.
- **Got:** Câu về tài trợ đào tạo 30 triệu và các cấp phê duyệt nghỉ phép.
- **Worst metric:** Answer relevancy = 0.4000.
- **Error Tree:** Output sai → context/ranking ưu tiên các đoạn cùng có “30 triệu” và “Giám đốc” → lexical reranker không hiểu ý định mua sắm.
- **Root cause:** Numeric overlap lấn át domain terms; câu hỏi multi-condition cần gom cùng lúc quy trình mua sắm, ngưỡng giá và xác nhận CNTT.
- **Suggested fix:** Boost metadata `source=mua_sam.md`, thêm query expansion “mua sắm/thiết bị/báo giá/cấu hình”, và rerank bằng cross-encoder thật.

### 2. Senior 9 năm: phép và lương

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép và 20–35 triệu VNĐ/tháng.
- **Got:** Trả đúng 18 ngày nhưng thiếu khoảng lương Senior.
- **Worst metric:** Context recall = 0.5455.
- **Error Tree:** Output thiếu → context chỉ đủ nhánh nghỉ phép → retriever không giữ chunk bảng lương trong top-3.
- **Root cause:** Multi-hop qua hai tài liệu (`nghi_phep_nam_v2024.md`, `bang_luong_2024.md`) nhưng top-k sau rerank thiên về cụm từ “ngày phép/thâm niên”.
- **Suggested fix:** Decompose thành hai sub-query “phép theo thâm niên” và “lương Senior”, sau đó hợp nhất context trước generation.

### 3. Lương thử việc Junior cao nhất

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** 85% × 20 triệu = 17 triệu VNĐ/tháng.
- **Got:** Chỉ nêu quy tắc 85%, không lấy mức trần Junior để tính.
- **Worst metric:** Answer relevancy = 0.5714.
- **Error Tree:** Output thiếu phép tính → context có quy tắc thử việc nhưng thiếu/không ưu tiên dòng Junior trong bảng lương.
- **Root cause:** Cần join bảng lương và chính sách thử việc, sau đó thực hiện phép tính xác định.
- **Suggested fix:** Table-aware chunking cho bảng lương, multi-query retrieval, và bước calculator có citation cho `0.85 × 20,000,000`.

### 4. Số ngày phép năm hiện hành

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** v2024 là 15 ngày; v2023 là 12 ngày và đã bị thay thế.
- **Got:** Đúng 15 ngày và quy tắc thâm niên nhưng chưa phát biểu rõ trạng thái superseded của v2023.
- **Worst metric:** Context recall = 0.6842.
- **Error Tree:** Output gần đúng → context có cả số cũ/mới → version relationship chưa được biểu diễn rõ trong metadata.
- **Root cause:** Hai tài liệu cùng match mạnh; enrichment chưa trích `effective_date`, `version`, `supersedes` thành trường lọc riêng.
- **Suggested fix:** Chuẩn hóa metadata phiên bản và filter ưu tiên `status=current`, nhưng vẫn lấy bản cũ làm context đối chiếu.

### 5. Chu kỳ đổi mật khẩu

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** v2.0 hiện hành là 120 ngày; v1.0 cũ là 90 ngày.
- **Got:** Nêu 90 ngày trước, rồi trộn yêu cầu 8 và 12 ký tự, không trả lời rõ 120 ngày hiện hành.
- **Worst metric:** Answer relevancy = 0.6000.
- **Error Tree:** Output sai phiên bản → context chứa hai phiên bản → reranker xem chúng gần như tương đương.
- **Root cause:** Lexical fallback không có temporal/version reasoning; extractive selector ưu tiên overlap thay vì tính hiện hành.
- **Suggested fix:** Metadata version filter, freshness boost, và prompt bắt buộc phát biểu “hiện hành” trước “đã thay thế”.

## Case Study

**Case:** Laptop 30 triệu là failure đại diện vì chứa ba điều kiện nằm trong một domain nhưng lexical similarity bị nhiễu bởi con số và chức danh xuất hiện ở domain khác.

1. Output đúng? → Không; thiếu báo giá và xác nhận cấu hình.
2. Context đúng? → Chưa; training/leave chunks chiếm chỗ procurement chunks.
3. Query rewrite ổn? → Chưa có decomposition hoặc domain expansion.
4. Fix ưu tiên → metadata filter `category=procurement`, cross-encoder thật, rồi kiểm tra đủ ba facts trước khi trả lời.

Nếu có thêm một giờ, ưu tiên schema metadata phiên bản/domain và query decomposition cho nhóm multi-hop/numeric trước khi tinh chỉnh prompt.

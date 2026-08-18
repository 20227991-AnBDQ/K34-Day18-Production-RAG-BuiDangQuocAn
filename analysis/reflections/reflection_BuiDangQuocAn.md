# Individual Reflection — Lab 18

**Tên:** Bùi Đặng Quốc An

**MSSV:** 2A202601799

**Phạm vi:** M1–M5 và pipeline tích hợp

## 1. Mapping bài giảng vào code

| Lecture concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 tạo 208 chunks (avg 99 ký tự), so với basic 51 (avg 410). Corpus policy có nhiều câu ngắn ít overlap nên threshold cần tune bằng retrieval eval, không chọn theo trực giác. |
| Parent-child + structure | M1 | `chunk_hierarchical()`, `chunk_structure_aware()` | Hierarchical tạo 101 child/11 parent khi đo trên corpus gộp; retrieve child tăng precision còn trả parent giữ đủ context. Structure-aware tạo 106 chunks và giữ header/section metadata. |
| BM25 + Dense fusion | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | RRF hợp nhất rank thay vì so trực tiếp hai thang điểm khác nhau. Production context recall proxy tăng từ 0.7422 lên 0.8785. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Thiết kế rerank top-20 xuống top-3. Local model stack lỗi nên dùng lexical fallback; query trung bình vẫn 13.03 ms nhưng các case version/multi-hop cho thấy chất lượng cross-encoder thật là cần thiết. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Tách faithfulness, answer relevancy, context precision và context recall giúp xác định failure nằm ở retrieval hay generation. Run hiện tại được gắn nhãn `lexical_proxy`, không giả làm RAGAS thật. |
| Contextual enrichment/HyQA | M5 | `_enrich_single_call()`, `enrich_chunks()` | Combined mode gom summary, questions, context và metadata trong một API call/chunk. Fallback vẫn tạo schema đầy đủ, giúp pipeline chạy khi thiếu key. |

## 2. Khó khăn và cách giải quyết

### PDF dependency

- **Exact error:** `ModuleNotFoundError: No module named 'pypdf'` tại `load_documents()`.
- **Debug:** Chạy toàn bộ pytest, xác định chỉ test compare strategies chạm PDF loader.
- **Giải pháp:** Cho PDF loader bỏ qua an toàn khi dependency chưa có, sau đó cài dependency khai báo. Kết quả đọc 26 tài liệu; hai PDF scan vẫn được cảnh báo cần OCR.

### Windows Unicode

- **Exact error:** `UnicodeEncodeError: 'charmap' codec can't encode characters ...` khi in cảnh báo tiếng Việt.
- **Debug:** Unit tests pass nhưng CLI fail trước chunking, nên nguyên nhân là stdout chứ không phải dữ liệu.
- **Giải pháp:** Reconfigure stdout/stderr UTF-8 trong shared config.

### Model/runtime và infrastructure

- **Exact error model:** `module 'ml_dtypes' has no attribute 'float8_e3m4'` khi Transformers kéo TensorFlow integration không cần thiết.
- **Exact error Qdrant:** `[WinError 10061] No connection could be made because the target machine actively refused it`.
- **Exact error Docker:** pipe `//./pipe/dockerDesktopLinuxEngine` không tồn tại.
- **Debug:** Kiểm tra riêng imports, model load, Docker Compose và pipeline.
- **Giải pháp:** Đặt `USE_TF=0` vì SentenceTransformers dùng PyTorch; cache model; cung cấp hashing embeddings/in-memory index và lexical reranker làm fallback có cảnh báo. Code vẫn giữ đường chạy BGE + Qdrant thật khi service/model sẵn sàng.

### Evaluation credentials

- **Exact error:** `OPENAI_API_KEY is not configured`.
- **Giải pháp:** Không gọi network ngầm; tạo lexical proxy có nhãn rõ ràng và per-question diagnostics. Khi có key, cùng hàm sẽ chạy RAGAS 4 metrics thật.

## 3. Action plan cho project

## Project: Trợ lý tra cứu chính sách nội bộ

### Hiện tại

- Pipeline: markdown/PDF text → hierarchical child chunks → enrichment → BM25 + dense → RRF → rerank → grounded answer → evaluation.
- Known issues: conflict phiên bản, multi-hop/table retrieval, scan PDF chưa OCR, Docker/model/API phụ thuộc môi trường.

### Plan áp dụng

1. [ ] **Chunking:** Giữ parent-child 2,048/256; thêm table-aware chunks và không nối qua ranh giới tài liệu.
2. [ ] **Search:** Hybrid BM25 + BGE-M3; thêm filters `source`, `category`, `version`, `effective_date`, `status`.
3. [ ] **Reranking:** BGE reranker top-20 → top-3; benchmark p50/p95 và cache model singleton.
4. [ ] **Evaluation:** Chạy RAGAS thật trong CI có secret, cộng exact-match cho numeric/version và bộ regression theo 6 question types.
5. [ ] **Enrichment:** Combined single-call; validate JSON schema, cache theo content hash, và track token/cost.
6. [ ] **Hard cases:** Query decomposition cho multi-hop; calculator cho numeric; OCR cho scan PDF; version-aware answer template.

### Timeline

- **Tuần 1:** Qdrant deployment, model cache, metadata schema và ingestion/OCR.
- **Tuần 2:** Hybrid retrieval, version filters, cross-encoder và query decomposition.
- **Tuần 3:** RAGAS + deterministic regression suite, failure dashboard và latency/cost budgets.
- **Tuần 4:** Load test, canary release, human review cho câu trả lời confidence thấp.

## 4. Tự đánh giá

| Tiêu chí | Điểm (1–5) | Bằng chứng |
|---|---:|---|
| Hiểu bài giảng | 5 | Mapping đủ 5 modules, nêu trade-off và failure modes |
| Code quality | 5 | Validation, caching, typed results, deterministic fallbacks |
| Problem solving | 5 | 37/37 tests, pipeline exit 0 trên môi trường thiếu services |
| Production thinking | 4 | Có latency/reporting và degradation; còn cần RAGAS thật + Docker |

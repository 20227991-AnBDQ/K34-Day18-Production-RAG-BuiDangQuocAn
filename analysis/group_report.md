# Báo cáo hoàn thành — Lab 18: Production RAG

**Hình thức:** Bài cá nhân

**Người thực hiện:** Bùi Đặng Quốc An

**MSSV:** 2A202601799

**Ngày:** 18/08/2026

## Phạm vi thực hiện

| Module | Nội dung | Trạng thái |
|---|---|---:|
| M1 | Semantic, hierarchical parent-child, structure-aware chunking | Hoàn thành |
| M2 | Vietnamese segmentation, BM25, dense/Qdrant, RRF | Hoàn thành |
| M3 | BGE cross-encoder reranking và lexical fallback | Hoàn thành |
| M4 | RAGAS integration, offline proxy, Diagnostic Tree | Hoàn thành |
| M5 | 4 enrichment techniques và combined single-call mode | Hoàn thành |
| Pipeline | Parent retrieval, grounded answer, reports, latency | Hoàn thành |

**Auto-tests:** 37/37 pass.

**Pipeline:** exit code 0 với in-memory dense fallback vì Docker Desktop chưa chạy.

## Kết quả đánh giá

| Metric | Naive | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer relevancy | 0.7587 | 0.8044 | +0.0458 |
| Context precision | 1.0000 | 1.0000 | +0.0000 |
| Context recall | 0.7422 | 0.8785 | +0.1363 |

Các số trên dùng backend `lexical_proxy` từ lần chạy offline; JSON ghi rõ backend để không nhầm với RAGAS thật.

## Latency breakdown

| Bước | Thời gian |
|---|---:|
| Chunking | 214.94 ms |
| Enrichment fallback (106 chunks) | 5.83 ms |
| Indexing + model/Qdrant fallback detection | 29,190.85 ms |
| Query trung bình | 13.03 ms |
| Query p95 | 13.48 ms |
| Evaluation proxy | 9.01 ms |

## Key findings

1. **Biggest improvement:** Context recall tăng 0.1363 nhờ hybrid retrieval và parent-child context.
2. **Biggest challenge:** Version conflict (v2023/v2024, v1/v2) và multi-hop qua nhiều tài liệu.
3. **Surprise finding:** Semantic threshold 0.85 tạo 208 chunks, nhiều hơn basic 51; threshold cao không mặc nhiên tốt và phải được tune trên corpus.
4. **Next optimization:** Thêm `version/status/effective_date` metadata, query decomposition, và chạy cross-encoder/RAGAS thật khi model và API key sẵn sàng.

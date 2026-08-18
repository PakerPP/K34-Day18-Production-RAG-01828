# Group Report — Lab 18: Production RAG

**Nhóm:** Cá nhân — Bùi Xuân Tùng (MSSV 2A202601828)
**Ngày:** 18/08/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Bùi Xuân Tùng | M1: Chunking | ☑ | 13/13 |
| Bùi Xuân Tùng | M2: Hybrid Search | ☑ | 5/5 |
| Bùi Xuân Tùng | M3: Reranking | ☑ | 5/5 |
| Bùi Xuân Tùng | M4: Evaluation | ☑ | 4/4 |
| Bùi Xuân Tùng | M5: Enrichment | ☑ | 10/10 |

**Tổng: 37/37 tests pass · 0 TODO còn lại trong `src/`**

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8500 | 0.8315 | −0.0185 |
| Answer Relevancy | 0.7217 | 0.7633 | +0.0416 |
| Context Precision | 0.9250 | 0.9417 | +0.0167 |
| Context Recall | 0.9250 | 0.8667 | −0.0583 |

*(n = 20 câu · cả 4 metric của Production đều ≥ 0.75)*

## Latency Breakdown

| Bước | Avg | Min | Max |
|------|-----|-----|-----|
| BM25 search | 0.82 ms | 0.76 ms | 0.96 ms |
| Dense search | 2265 ms* | 100 ms | 10891 ms* |
| RRF fusion | 0.05 ms | 0.04 ms | 0.06 ms |
| Cross-encoder rerank | 5901 ms | 4787 ms | 7775 ms |
| **Tổng retrieval** | **~8167 ms** | | |

*\* Lần query đầu tiên gồm cả thời gian load model bge-m3; các lần sau ổn định ~100 ms.*
*(5 queries · 104 chunks · CPU · BM25 index build: 27.4 s cho toàn corpus)*

## Key Findings

1. **Biggest improvement:** **Answer Relevancy +0.0416** (0.7217 → 0.7633) — metric duy nhất dưới ngưỡng ở baseline, và cũng là metric cải thiện nhiều nhất. Context Precision cũng tăng (+0.0167) lên 0.9417, phản ánh việc cross-encoder rerank lọc top-20 xuống top-3 giúp context sạch hơn.

2. **Biggest challenge:** RAGAS trả về cả 4 metric = 0.0000 mà không báo lỗi. Nguyên nhân thật là `AuthenticationError 401` (API key hỏng) nhưng bị `except Exception` nuốt mất, khiến điểm 0 do lỗi credential trông y hệt điểm 0 do chất lượng kém. Debug bằng cách cô lập xuống 1 câu hỏi để log ngắn lại và lộ ra exception thật.

3. **Surprise finding:** **Production không thắng Naive ở mọi mặt.** Context Recall *giảm* 0.0583 và Faithfulness giảm nhẹ 0.0185. Lý do: `RERANK_TOP_K = 3` cắt context xuống còn 3 chunk, đổi recall lấy precision — trong khi baseline giữ nguyên 3 chunk dense nhưng mỗi chunk là paragraph lớn (chunk_basic 500 chars) nên chứa nhiều thông tin hơn child chunk 256 chars. Trên corpus nhỏ và sạch như thế này, baseline vốn đã rất mạnh (0.925 recall), nên phần "cải thiện" mà pipeline production mang lại chủ yếu nằm ở khả năng trả lời đúng trọng tâm chứ không phải ở lượng thông tin lấy về.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** Baseline đã mạnh sẵn (3/4 metric ≥ 0.85) vì corpus nhỏ, sạch, tiếng Việt thuần. Production đưa cả 4 metric lên ≥ 0.75, thắng ở Answer Relevancy và Context Precision, đổi lại mất một ít Context Recall.

2. **Biggest win — module nào, tại sao:** M3 Reranking. Cross-encoder chấm lại từng cặp (query, doc) nên đẩy Context Precision lên 0.9417. Nhưng phải trả giá: ~5.9 s/query trên CPU, chiếm 72% tổng thời gian retrieval. Đây chính là lý do kiến trúc bắt buộc phải là *retrieve rộng top-20 → rerank hẹp top-3*, không thể rerank toàn bộ corpus.

3. **Case study — 1 failure, Error Tree walkthrough:** Câu *"Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?"*
   - Output đúng? → KHÔNG (trả "Không tìm thấy.")
   - Context đúng? → **CÓ, hoàn hảo** — precision 1.0, recall 1.0
   - Query OK? → CÓ
   - → Fix ở bước **Generation**, không phải retrieval.

   Đây là câu hỏi phủ định: đáp án đúng là "KHÔNG", diễn đạt gián tiếp trong tài liệu (liệt kê ai *được* hưởng, thử việc không có trong danh sách). LLM không suy ra phủ định từ sự vắng mặt nên bỏ cuộc. 3/5 failure tệ nhất đều có context đúng — nút thắt của pipeline là generation, không phải search.

4. **Next optimization nếu có thêm 1 giờ:**
   - Sửa prompt: thêm hướng dẫn xử lý câu hỏi phủ định, yêu cầu dẫn lại điều kiện áp dụng thay vì trả lời cộc lốc. Rẻ nhất, tác động tới 3/5 failure.
   - Tăng `RERANK_TOP_K` 3 → 5 cho câu hỏi đa mệnh đề, đo xem đổi bao nhiêu precision lấy recall.
   - Thêm metadata phiên bản/ngày hiệu lực để lọc chính sách đã bị thay thế (corpus có v2023 vs v2024, v1 vs v2 nằm chung index).

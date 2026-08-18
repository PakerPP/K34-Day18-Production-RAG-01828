# Reflection — Lab 18: Production RAG Pipeline

**Tên:** Bùi Xuân Tùng — MSSV 2A202601828

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Cắt chunk khi cosine giữa 2 câu liền kề < threshold. Với threshold mặc định 0.85 trên corpus tiếng Việt, hầu như câu nào cũng bị tách (embedding `all-MiniLM-L6-v2` cho similarity giữa các câu tiếng Việt thường dưới 0.85) → ra rất nhiều chunk vụn. Threshold là tham số phải tune theo ngôn ngữ + model, không dùng chung một con số cho mọi corpus. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent 2048 / child 256 trên corpus 26 documents → 26 parents, 104 children (~4 child/parent). Child nhỏ giúp match chính xác, parent giữ ngữ cảnh. Mỗi child mang `parent_id` để truy ngược về parent. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Dùng capturing group trong `re.split(r'(^#{1,3}\s+.+$)')` để **giữ lại** header trong kết quả split. Ưu điểm rõ trên corpus này: bảng lương trong `bang_luong_2024.md` nằm gọn trong 1 chunk cùng header của nó, không bị cắt ngang giữa bảng. |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | `underthesea` nối từ ghép bằng `_` ("nghỉ_phép"). Bắt buộc `replace("_", " ")`, nếu không document có token `nghỉ_phép` còn query có `nghỉ` + `phép` → BM25 không khớp token nào, search trả rỗng. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF chỉ dùng **thứ hạng**, không dùng điểm số, nên gộp được BM25 (điểm không chặn trên) với cosine (0–1) mà không cần normalize. Công thức `1/(k + rank + 1)`, k=60. Document được cả 2 retriever xếp cao mới lên đầu. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` chấm lại từng cặp (query, doc) thay vì so vector độc lập → chính xác hơn nhưng đắt: đo được ~30s cho 20 documents trên CPU. Chính chi phí này quyết định kiến trúc retrieve top-20 → rerank top-3. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Metric thấp nhất là **answer_relevancy** (≈0.72). Nguyên nhân không phải retrieval: nhiều câu có `context_recall` = 1.0 (context đúng đã được lấy về) nhưng LLM vẫn trả lời "Không tìm thấy." → điểm relevancy = 0. Đây là vấn đề prompt, không phải vấn đề search. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Prepend 1 câu mô tả chunk nằm ở đâu trong tài liệu trước khi embed. Giúp chunk mồ côi (ví dụ đoạn chỉ ghi "tối đa 30 ngày") vẫn mang thông tin về chủ đề và văn bản gốc, giảm khả năng retrieval trượt. Dùng combined mode: 1 API call/chunk cho cả summary + questions + context + metadata. |

## Phần 2: Khó khăn & giải quyết

**Khó khăn 1 — RAGAS trả về cả 4 metric = 0.0000 mà không báo lỗi.**

Exact error message (chỉ hiện khi chạy với `PYTHONUNBUFFERED=1`):
```
Exception raised in Job[2]: AuthenticationError(Error code: 401 -
{'error': {'message': 'Incorrect API key provided: sk-proj-****YfQA',
'type': 'invalid_request_error', 'code': 'invalid_api_key'}, 'status': 401})
```

Cách debug:
1. Đọc `ragas_report.json`: `num_questions: 20` nhưng mọi metric = 0 → RAGAS **có** chạy và trả đủ 20 dòng, chỉ là mỗi dòng đều NaN. Loại trừ khả năng sai ở khâu dựng `Dataset`.
2. Cô lập: gọi `evaluate_ragas()` với đúng 1 câu hỏi thay vì cả test set → log ngắn lại, lộ ra `AuthenticationError 401`.
3. Xác nhận nguyên nhân gốc: gọi OpenAI trần không qua RAGAS → cũng 401. Kết luận là lỗi credential chứ không phải lỗi code.
4. Thay API key hợp lệ → faithfulness 0.85, context_precision 0.925.

Kiến thức thiếu → cách bổ sung: tôi chưa nắm được RAGAS xử lý lỗi per-job như thế nào (nó bắt exception từng job rồi điền NaN thay vì raise ra ngoài). Đọc lại cách RAGAS gom kết quả để hiểu vì sao lỗi không nổi lên tới caller.

**Khó khăn 2 — `UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3`** khi `load_documents()` in cảnh báo tiếng Việt + emoji. Nguyên nhân: console Windows mặc định cp1252. Giải quyết bằng biến môi trường `PYTHONIOENCODING=utf-8` khi chạy, không sửa file scaffold.

**Bài học chung:** `except Exception` bao quanh RAGAS (theo đúng hướng dẫn trong TODO) giúp pipeline không crash, nhưng đồng thời nuốt mất thông báo lỗi thật. Điểm 0 do hỏng credential trông y hệt điểm 0 do chất lượng trả lời kém. Lần sau nên in `type(e).__name__` + message trước khi trả về zeros.

## Phần 3: Action Plan cho project

## Project: Trợ lý hỏi–đáp tài liệu nội bộ (RAG tiếng Việt)

### Hiện tại
- RAG pipeline hiện tại: chunk theo đoạn văn cố định → embed → dense search top-k → đưa thẳng vào LLM. Không có BM25, không rerank, không đánh giá định lượng.
- Known issues:
  - Câu hỏi chứa số hiệu / mã văn bản / thuật ngữ hiếm thường trượt, vì dense search không khớp tốt token hiếm.
  - Tài liệu có nhiều phiên bản (v2023 vs v2024) → hệ thống trả lời lẫn lộn giữa bản cũ và bản hiện hành. Lab này tái hiện đúng vấn đề đó: câu "nhân viên 9 năm thâm niên được bao nhiêu ngày phép" nhận câu trả lời trộn cả 15 ngày (v2024) lẫn 12 ngày (v2023).
  - Không biết pipeline tốt hay tệ vì chưa có metric nào.

### Plan áp dụng
1. [ ] **Chunking strategy:** Structure-aware làm chính (tài liệu nội bộ đều là markdown/docx có heading rõ), fallback hierarchical cho file không có cấu trúc. Lý do: giữ được bảng và mục nguyên vẹn — đã thấy hiệu quả rõ với bảng lương trong lab.
2. [ ] **Search:** Hybrid BM25 + Dense với RRF. Lý do: BM25 bắt được mã văn bản và thuật ngữ hiếm mà dense hay trượt; RRF gộp mà không cần normalize điểm.
3. [ ] **Reranking:** Có, dùng `bge-reranker-v2-m3`. Nhưng phải chạy trên GPU hoặc đổi sang FlashRank — đo trên CPU ~30s/query là không dùng được cho hệ thống có người dùng thật.
4. [ ] **Evaluation:** RAGAS 4 metrics làm chuẩn, chạy trên test set ~30 câu tự xây, bám theo các loại câu hỏi thật của người dùng (tra cứu, phủ định, đa bước, xung đột phiên bản). Kèm `failure_analysis()` theo Diagnostic Tree để biết nên sửa ở khâu nào.
5. [ ] **Enrichment:** Contextual prepend (combined mode, 1 API call/chunk). Ưu tiên kỹ thuật này vì rẻ, không cần đổi kiến trúc retrieval, và giải quyết đúng vấn đề chunk mồ côi.

**Ưu tiên bổ sung rút ra từ lab:** thêm metadata phiên bản/ngày hiệu lực vào từng chunk và lọc theo đó trước khi trả context. Trong lab, mọi bản chính sách cũ và mới đều nằm chung một index nên retrieval kéo cả hai lên; không có tín hiệu phiên bản thì LLM không có cách nào biết bản nào còn hiệu lực.

### Timeline
- **Tuần 1:** Dựng test set 30 câu + đo baseline hiện tại bằng RAGAS. Không sửa gì cho tới khi có số.
- **Tuần 2:** Đổi sang structure-aware chunking + thêm metadata phiên bản. Đo lại.
- **Tuần 3:** Thêm BM25 + RRF vào pipeline dense sẵn có. Đo lại, so với tuần 2.
- **Tuần 4:** Thêm reranking (benchmark FlashRank vs bge-reranker về latency trước khi chọn). Đo lại.
- **Tuần 5:** Enrichment contextual prepend cho toàn bộ corpus. Đo lần cuối, chốt cấu hình theo bảng so sánh 4 mốc.

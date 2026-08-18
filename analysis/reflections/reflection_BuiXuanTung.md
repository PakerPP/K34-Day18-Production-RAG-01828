# Individual Reflection — Lab 18

**Tên:** Bùi Xuân Tùng — MSSV 2A202601828
**Module phụ trách:** M1 + M2 + M3 + M4 + M5 (bài cá nhân — implement toàn bộ)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: **M1, M2, M3, M4, M5** (toàn bộ 5 module, 0 TODO còn lại)
- Các hàm/class chính đã viết:

| Module | Hàm/Class |
|--------|-----------|
| M1 | `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, helper `_split_by_size()`, `_get_semantic_encoder()` |
| M2 | `segment_vietnamese()`, `BM25Search.index()/.search()`, `DenseSearch.index()/.search()`, `reciprocal_rank_fusion()` |
| M3 | `CrossEncoderReranker._load_model()/.rerank()`, `FlashrankReranker._load_model()/.rerank()` |
| M4 | `evaluate_ragas()`, `failure_analysis()` |
| M5 | `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()`, `_enrich_single_call()` |

- Số tests pass: **37 / 37** (`pytest tests/ -v`)

## 2. Kiến thức học được

- Khái niệm mới nhất: **RRF (Reciprocal Rank Fusion)**. Điều làm tôi thấy hay là RRF chỉ dùng *thứ hạng* chứ không dùng *điểm số* — nhờ vậy mới gộp được BM25 (điểm không chặn trên, thang tùy corpus) với cosine similarity (thang 0–1) mà không cần normalize. Công thức `1/(k + rank + 1)` với k=60 làm điểm giữa các rank gần nhau, nên một document phải được **cả hai** retriever xếp hạng cao mới lên đầu.

- Điều bất ngờ nhất: **hai bước "phụ" lại quyết định kết quả nhiều hơn thuật toán chính**.
  1. `underthesea` nối từ ghép bằng dấu `_` ("nghỉ_phép"). Nếu không `replace("_", " ")` thì document tokenize thành `["nghỉ_phép"]` còn query tokenize thành `["nghỉ", "phép"]` → BM25 không khớp một token nào, search trả rỗng dù dữ liệu đúng nằm đó.
  2. Cross-encoder chấm lại từng cặp (query, document) nên chính xác hơn hẳn bi-encoder, nhưng đắt: đo được ~30s cho 20 documents trên CPU. Đây chính là lý do kiến trúc phải là *retrieve rộng (top-20) → rerank hẹp (top-3)*, không thể rerank toàn bộ corpus.

- Kết nối với bài giảng: phần Hybrid Search & Fusion (RRF), phần Reranking (cross-encoder vs bi-encoder), phần RAGAS 4 metrics và Diagnostic Tree cho failure analysis.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất: RAGAS trả về cả 4 metric = 0.0000 mà không báo lỗi gì.**

  Exact error message (chỉ thấy khi bật `PYTHONUNBUFFERED=1`):
  ```
  Exception raised in Job[2]: AuthenticationError(Error code: 401 -
  {'error': {'message': 'Incorrect API key provided: sk-proj-****YfQA',
  'type': 'invalid_request_error', 'code': 'invalid_api_key'}, 'status': 401})
  ```

- **Cách giải quyết:** Ban đầu tôi tưởng `evaluate_ragas()` viết sai. Cách debug:
  1. Đọc `ragas_report.json` thấy `num_questions: 20` nhưng mọi metric = 0 → chứng tỏ RAGAS **có** chạy và **có** trả về 20 dòng, chỉ là mỗi dòng đều NaN. Vậy lỗi không nằm ở khâu tạo Dataset.
  2. Cô lập bằng cách gọi `evaluate_ragas()` với đúng 1 câu hỏi thay vì cả test set → log ngắn lại, lộ ra `AuthenticationError 401`.
  3. Xác nhận nguyên nhân gốc bằng một call OpenAI trần (không qua RAGAS) → cũng 401. Kết luận: lỗi credential, không phải lỗi code.
  4. Sau khi thay API key hợp lệ: faithfulness 0.85, context_precision 0.925.

  Bài học rút ra: `except Exception` bao quanh RAGAS (theo đúng hướng dẫn TODO) giúp pipeline không crash, nhưng cũng **nuốt mất thông báo lỗi thật**. Điểm 0 do hỏng credential trông y hệt điểm 0 do model trả lời tệ. Lần sau tôi sẽ in `type(e).__name__` kèm message ra stderr trước khi trả về zeros.

- **Khó khăn 2:** `UnicodeEncodeError: 'charmap' codec can't encode characters` khi `load_documents()` in cảnh báo tiếng Việt + emoji. Nguyên nhân: console Windows mặc định cp1252. Giải quyết bằng `PYTHONIOENCODING=utf-8` khi chạy, không sửa file scaffold.

- Thời gian debug: khoảng 45 phút, phần lớn cho vụ RAGAS = 0.

## 4. Nếu làm lại

- **Sẽ làm khác:** Test từng module bằng dữ liệu thật (corpus `data/`) ngay sau khi viết, thay vì chỉ chạy `pytest`. Test có sẵn dùng corpus 2–3 câu nên không phản ánh hành vi trên corpus thật; có trường hợp `pytest` xanh hết nhưng chạy trên dữ liệu thật mới lộ vấn đề.
- **Module muốn thử tiếp:** M5 Enrichment. Hiện `enriched_text` mới chỉ là contextual prepend; tôi muốn thử index thêm cả `summary` và `hypothesis_questions` như vector riêng để đo xem HyQA có thực sự thu hẹp khoảng cách từ vựng giữa câu hỏi người dùng và văn bản chính sách hay không.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | — (bài cá nhân) |
| Problem solving | 4 |

# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Cá nhân
**Thành viên:** Bùi Xuân Tùng (MSSV 2A202601828) → M1 · M2 · M3 · M4 · M5

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8500 | 0.8315 | −0.0185 |
| Answer Relevancy | 0.7217 | 0.7633 | +0.0416 |
| Context Precision | 0.9250 | 0.9417 | +0.0167 |
| Context Recall | 0.9250 | 0.8667 | −0.0583 |

*(n = 20 câu · Naive = paragraph chunking + dense-only · Production = hierarchical chunking + enrichment + hybrid BM25/dense + RRF + cross-encoder rerank top-3)*

## Bottom-5 Failures

### #1
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5–50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Cần xác nhận cấu hình kỹ thuật từ phòng CNTT, và đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness = 0.0 (answer_relevancy 0.0, context_recall 0.3333, context_precision 1.0)
- **Error Tree:** Output sai → Context đúng? → **Chỉ đúng một phần** (context_recall 0.33: lấy được ngưỡng phê duyệt nhưng thiếu quy định 3 báo giá và xác nhận CNTT) → Query OK? → Có, nhưng là câu hỏi đa bước hỏi 2 việc trong 1 câu
- **Root cause:** Câu hỏi multi-hop cần ghép 3 mảnh thông tin nằm ở 3 chunk khác nhau (ngưỡng duyệt theo giá trị, quy định báo giá, quy định thiết bị CNTT). `RERANK_TOP_K = 3` chỉ giữ lại 3 chunk, không đủ để phủ hết. Thiếu mảnh → LLM theo đúng chỉ thị "chỉ trả lời dựa trên context" nên chọn trả lời "Không tìm thấy" thay vì trả lời một nửa.
- **Suggested fix:** Tăng `RERANK_TOP_K` lên 5 cho câu hỏi dài/đa mệnh đề, hoặc tách câu hỏi thành các sub-query rồi gộp context (query decomposition).

### #2
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16–30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** "Giám đốc điều hành (CEO)."
- **Worst metric:** faithfulness = 0.0 (answer_relevancy 0.786, context_recall 0.5, context_precision 1.0)
- **Error Tree:** Output sai → Context đúng? → **Đúng một nửa** (context_recall 0.5: có bảng phân cấp phê duyệt, thiếu đoạn về nghĩa vụ bảo hiểm) → Query OK? → Có
- **Root cause:** Câu trả lời về mặt nội dung là **đúng**, nhưng quá cộc lốc: chỉ 1 cụm danh từ, không dẫn lại điều kiện "16–30 ngày" từ context. RAGAS faithfulness kiểm tra từng câu (claim) trong answer có truy vết được về context không — một câu trả lời trần trụi không có cấu trúc claim rõ ràng nên bị chấm 0.
- **Suggested fix:** Sửa prompt yêu cầu trả lời đầy đủ mệnh đề, dẫn lại điều kiện áp dụng (VD "Nghỉ 16–30 ngày cần CEO phê duyệt") thay vì trả lời cụt.

### #3
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** "Không tìm thấy."
- **Worst metric:** answer_relevancy = 0.0 (faithfulness 1.0, context_precision 1.0, **context_recall 1.0**)
- **Error Tree:** Output sai → Context đúng? → **ĐÚNG HOÀN TOÀN** (cả precision lẫn recall = 1.0) → Query OK? → Có → **Fix ở bước Generation**
- **Root cause:** Đây là failure rõ ràng nhất của lab. Retrieval hoạt động hoàn hảo — chunk chứa đáp án đã nằm trong context. Nhưng đây là **câu hỏi phủ định**: đáp án đúng là "KHÔNG", và thông tin đó được diễn đạt gián tiếp trong tài liệu (liệt kê ai *được* hưởng, thử việc không có trong danh sách). LLM không suy luận được phủ định từ sự vắng mặt, nên bỏ cuộc và trả "Không tìm thấy."
- **Suggested fix:** Thêm hướng dẫn xử lý câu hỏi phủ định vào prompt: "Nếu context liệt kê đối tượng áp dụng mà không bao gồm đối tượng được hỏi, hãy trả lời KHÔNG và giải thích." Không cần đụng tới retrieval.

### #4
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** "Nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ. Vì vậy, nhân viên sẽ phải hoàn trả **25.000.000 VNĐ**."
- **Worst metric:** faithfulness = 0.5 (answer_relevancy 0.784, context_precision 1.0, context_recall 1.0)
- **Error Tree:** Output gần đúng → Context đúng? → Có (recall 1.0) → Query OK? → Có → Fix ở bước Generation
- **Root cause:** Con số cuối cùng **đúng**, nhưng LLM bỏ qua bước lập luận trung gian: không nêu ra điều kiện "cam kết 1 năm" và không nói rõ vì sao 8 tháng < 12 tháng nên mới phải trả 100%. Faithfulness bị trừ vì claim "phải hoàn trả 100%" xuất hiện mà không kèm điều kiện làm nó đúng.
- **Suggested fix:** Prompt yêu cầu nêu rõ căn cứ trước khi đưa kết luận số học ("Theo quy định X... vì vậy...").

### #5
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** "Bị phạt 2%/tháng trên số tiền chưa hoàn ứng... Sau 20 ngày, khoản tạm ứng sẽ bị tính phí cho 5 ngày quá hạn (20 − 15)..."
- **Worst metric:** faithfulness = 0.7143 (answer_relevancy 0.794, context_precision 0.8333, context_recall 1.0)
- **Error Tree:** Output gần đúng → Context đúng? → Có (recall 1.0) → Query OK? → Có → Fix ở bước Generation
- **Root cause:** Lập luận đúng hướng (nhận ra quá hạn 5 ngày, đúng công thức 2%/tháng) nhưng LLM tự thực hiện phép tính pro-rata mà context không quy định cách tính — phần suy diễn số học này không truy vết được về tài liệu nên faithfulness bị trừ. Đây là ranh giới giữa "trả lời dựa trên context" và "tự tính toán".
- **Suggested fix:** Với câu hỏi tính toán, yêu cầu LLM nêu công thức lấy từ context và ghi rõ phần nào là suy diễn thêm; hoặc tách phần tính toán ra khỏi phần trích dẫn.

## Case Study (cho presentation)

**Question chọn phân tích:** *"Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?"* (Failure #3)

Chọn câu này vì nó tách bạch hoàn hảo giữa lỗi retrieval và lỗi generation — điều mà 4 metric của RAGAS sinh ra chính là để phân biệt.

**Error Tree walkthrough:**
1. **Output đúng?** → KHÔNG. Trả về "Không tìm thấy." trong khi đáp án đúng là "KHÔNG, thử việc chưa được hưởng PVI."
2. **Context đúng?** → **CÓ, hoàn hảo.** `context_precision = 1.0` và `context_recall = 1.0`. Toàn bộ thông tin cần thiết đã nằm trong 3 chunk được đưa cho LLM. Hybrid search + rerank đã làm đúng việc của nó.
3. **Query rewrite OK?** → CÓ. Câu hỏi rõ ràng, không mơ hồ, không cần viết lại.
4. **Fix ở bước:** **Generation (prompt)**, không phải retrieval.

**Kết luận rút ra:** nếu chỉ nhìn điểm tổng thì dễ kết luận nhầm là "cần cải thiện search". Nhưng `context_recall = 1.0` chứng minh search đã đúng — vấn đề nằm ở chỗ LLM không suy luận được phủ định từ sự vắng mặt trong danh sách. Đổ thêm công sức vào chunking hay reranking cho câu này sẽ **không cải thiện gì**.

Đây cũng là lý do 3 trong 5 failure tệ nhất đều có `context_recall ≥ 0.5` và `context_precision = 1.0`: nút thắt của pipeline hiện tại là **generation**, không phải retrieval.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Viết lại prompt sinh câu trả lời: thêm hướng dẫn xử lý câu hỏi phủ định (#3), yêu cầu dẫn lại điều kiện áp dụng thay vì trả lời cộc lốc (#2, #4). Đây là thay đổi rẻ nhất và tác động tới 3/5 failure.
- Tăng `RERANK_TOP_K` từ 3 lên 5 cho câu hỏi đa mệnh đề (#1) và đo lại xem context_precision giảm bao nhiêu để đổi lấy recall.
- Thêm metadata phiên bản/ngày hiệu lực vào chunk để lọc bản chính sách đã bị thay thế trước khi đưa vào context.

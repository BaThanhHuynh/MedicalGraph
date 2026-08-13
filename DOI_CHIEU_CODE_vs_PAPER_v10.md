# Đối chiếu code MedicalGraph với Paper_Medical_PHT_v10

Ngày 13/08/2026. Đối tượng: mã nguồn khôi phục tại `D:\codevstdio\MedicalGraph`
so với `Paper_Medical_PHT_v10.docx` và gói số liệu `solieu_yte_new/`.

## Kết luận ngắn

Repo khôi phục **không chứa hệ thống được đo trong bài báo**. Nó là một thế hệ
trước đó — schema khác, mô hình khác, không có bộ điều khiển ba nhánh, không có
tái xếp hạng BGE. Toàn bộ số liệu Bảng 7–10 đến từ `medkgbert_dx.py`, tệp không
hề có trong repo (chỉ còn trên Drive, đã tải về `C:\BACKUP_RESEARCH_2026-08-11\tu_drive_2026-08-12\code\`).

Vì vậy việc "tinh chỉnh cho khớp bài báo" không phải là sửa tham số trong repo,
mà là **đưa thế hệ of-record vào repo** rồi chỉnh cấu hình theo Phụ lục D.

---

## 1. Hai thế hệ đồ thị

| | Thế hệ A — repo khôi phục | Thế hệ B — of-record (bài báo) |
|---|---|---|
| Script dựng | `code/database/import_neo4j.py` | `code/database/import_neo4j_v2.py` |
| Nguồn | `data/json/ICD10_LLM.json` (1.096 bản ghi) | `data/medical_data_cleaned.csv` (1.109 bản ghi) |
| Nhãn node | `Disease`, `Symptom`, `ICD10` | `BenhLy`, `TrieuChung`, `ICD10`, `ChuyenKhoa`, `XetNghiem` |
| Quan hệ | `HAS_SYMPTOM`, `HAS_NEG_SYMPTOM`, `MAPPED_TO` | `CO_TRIEU_CHUNG`, `LOAI_TRU_TRIEU_CHUNG`, `CO_MA_ICD10`, `THUOC_CHUYEN_KHOA`, `DUOC_CHAN_DOAN_BANG` |
| Khóa triệu chứng | `name` | `ten_trieu_chung` |
| Tách triệu chứng | xuống dòng `\n` | dấu `\|`, ưu tiên `positive_symptoms_norm` |

Bài báo Mục 3.1 đặc tả **5 lớp thực thể và 5 loại quan hệ**. Thế hệ A chỉ có 3 và 3.
Mọi truy vấn Cypher trong `medkgbert_dx.py` đều dùng nhãn tiếng Việt, nên chạy trên
đồ thị thế hệ A sẽ trả về rỗng — không phải lỗi, mà là sai schema.

### Quy mô đồ thị

| Thành phần | Thế hệ A | Bài báo (Bảng 3) | Lệch |
|---|---|---|---|
| BenhLy / Disease | 1.096 | 1.109 | −13 |
| TrieuChung / Symptom | 7.559 | 7.559 | 0 |
| ICD10 | 865 | 865 | 0 |
| ChuyenKhoa | — | 401 | **thiếu hẳn** |
| XetNghiem | — | 17 | **thiếu hẳn** |
| **Tổng thực thể E** | **9.520** | **9.951** | −431 |
| CO_TRIEU_CHUNG | 12.739 | 12.923 | −184 |
| LOAI_TRU_TRIEU_CHUNG | 1.767 | 1.091 | +676 |
| CO_MA_ICD10 | 1.096 | 1.096 | 0 |
| THUOC_CHUYEN_KHOA | — | 1.036 | **thiếu hẳn** |
| DUOC_CHAN_DOAN_BANG | — | 2.044 | **thiếu hẳn** |
| **Tổng bộ ba T** | **15.602** | **18.190** | −2.588 |

Đọc bảng này: phần lõi bệnh–triệu chứng–ICD của hai thế hệ gần như trùng nhau
(TrieuChung và ICD10 khớp tuyệt đối), nên nội dung trích xuất là cùng một đợt.
Khác biệt nằm ở **cách dựng đồ thị**: thế hệ A bỏ hẳn hai lớp ChuyenKhoa và
XetNghiem cùng 3.080 bộ ba của chúng.

Chênh lệch 13 bệnh chính là 13 node cô lập mà `structure_report.json` ghi nhận
(`BenhLy.isolated = 13`) — các bệnh không có mã ICD nên bị thế hệ A loại.
Chênh lệch `LOAI_TRU_TRIEU_CHUNG` ngược dấu vì hai thế hệ đọc hai cột khác nhau
(`negated_symptoms` trong JSON so với `negative_symptoms` trong CSV).

### Kiểm chứng: dựng lại Bảng 3 từ nguồn

Chạy lại logic của `import_neo4j_v2.py` trên `medical_data_cleaned.csv`:

| Chỉ số | Bài báo | Dựng lại | Lệch |
|---|---|---|---|
| BenhLy | 1.109 | 1.109 | 0 |
| TrieuChung | 7.559 | 7.558 | −1 |
| ICD10 | 865 | 865 | 0 |
| ChuyenKhoa | 401 | 401 | 0 |
| XetNghiem | 17 | 17 | 0 |
| Cả 5 loại quan hệ | 18.190 | **18.190** | **0** |

Toàn bộ 18.190 bộ ba tái lập chính xác. Lệch 1 node TrieuChung là node mồ côi đã
được chính `structure_report.json` ghi nhận (`TrieuChung.isolated = 1`;
`with_embedding_768d = 7558/7559`) — node này do `embed_constractive.py` tạo từ
cột raw mà không qua bộ lọc độ dài 2–100 của script import. Không ảnh hưởng số liệu.

**Nghĩa là:** `import_neo4j_v2.py` + `medical_data_cleaned.csv` đủ để dựng lại
đúng đồ thị của bài báo. Đây là phần đã khôi phục trọn vẹn.

---

## 2. Hai thế hệ hệ thống truy xuất

`code/rag/RAG.py` trong repo là một chatbot GraphRAG hoàn toàn khác:

| | `RAG.py` (thế hệ A) | Bài báo / `medkgbert_dx.py` |
|---|---|---|
| Sinh câu trả lời | Gemini qua LangChain | không sinh; chỉ xếp hạng ứng viên |
| Nhúng | PhoBERT gốc, vector `[CLS]`, `max_length=256` | PhoBERT tinh chỉnh tương phản, SentenceTransformer, 768 chiều chuẩn hóa |
| Truy xuất | `Neo4jVector` similarity, `k=15` | FAISS dense + BM25 → RRF (k=60) |
| Kênh đồ thị | không có | 4 tín hiệu Cypher (vector index, bigram, unigram, symptom CONTAINS) |
| Tái xếp hạng | không có | BGE-reranker-v2-m3 |
| Bộ điều khiển | không có | 3 nhánh theo τ_high / τ_med |
| Chunking | 400 / overlap 50 | không dùng |

Không có tham số nào của `RAG.py` ánh xạ được sang bài báo. Đây là ứng dụng minh họa,
không phải hệ thống được đo. Tương tự, `Faiss/custom_phobert_faiss.idx` được dựng
bằng PhoBERT gốc nên **không dùng lại được** với bộ mã hóa tương phản.

---

## 3. Đối chiếu tham số `medkgbert_dx.py` với Phụ lục D

| Phụ lục D | Giá trị bài báo | Code (sau chỉnh) | Trạng thái |
|---|---|---|---|
| Retrieval Depth TOP_K | 10 | `TOP_K = 10` | khớp |
| RRF Constant k | 60 | `RRF_K = 60` | khớp |
| RERANK_TOP_N | 5 | `RERANK_TOP_N = 5` | khớp |
| τ_high | 0.85 | `KG_HIGH_CONF_THRESHOLD = 0.85` | khớp |
| τ_med | 0.65 | `KG_MED_CONF_THRESHOLD = 0.65` | khớp |
| Consensus Bonus | 0.40 | `CONSENSUS_BONUS = 0.40` | khớp |
| Rerank Max Window (VN) | 1024 | `BGE_MAX_LENGTH = 1024` | khớp |
| KG_SYMP_MAX | 6 | `KG_SYMP_MAX = 6` | khớp |

Các trọng số hợp nhất cũng khớp từng hệ số:

| Phương trình | Bài báo | Code |
|---|---|---|
| (11) Consensus | `0,75 sKG + 0,25 sRAG + 0,40` | `ALPHA_CONSENSUS=0.25` → `0.25·rag + 0.75·kg`, bonus `0.40` |
| (13) KG High Conf | `0,45 sKG + 0,55 σ(rerank) + 0,20` | `0.55·bge + 0.45·kg`, `+0.20` nếu ∈ Top-5 RAG |
| (20) Disagree | `wKG=wrr=0,40; wRAG=0,20; bco=0,30; λ=0,15` | `0.40·bge + 0.40·kg + 0.20·rag`, `+0.30`, `+0.15` |

Ba phương trình hợp nhất khớp hoàn toàn. Hai điểm lệch được xử lý bên dưới.

### 3.1. Đã sửa: bộ tái xếp hạng LLM bật mặc định

`USE_LLM_CROSS_RERANK` mặc định là `True`. Nhưng Mục 4 viết cấu hình báo cáo
"không sử dụng thành phần mô hình ngôn ngữ lớn tái xếp hạng đa phương thức",
và ghi chú Bảng 7 nói rõ "bỏ bộ chọn Llama-3-8B ở các đường cơ sở".

Chạy code với mặc định cũ sẽ cho số **khác** Bảng 7. Đã đổi mặc định thành `False`
để tái lập đúng lượt đối xứng.

### 3.2. Cần bổ sung vào bài: điều kiện số ứng viên

Phương trình (15) định nghĩa nhánh KG High Confidence chỉ theo hai điều kiện:
`d_KG ≠ d_RAG` và `C_KG ≥ τ_high`. Code of-record có thêm điều kiện thứ ba:

```python
if kg_conf >= KG_HIGH_CONF_THRESHOLD and len(kg_ranked) >= 3:
```

Ràng buộc `len(kg_ranked) >= 3` không xuất hiện ở đâu trong bài. Đây là hành vi
thật đã sinh ra con số 87,67%, nên **không được xóa** — xóa đi là đổi kết quả.
Đã tách thành hằng số có tên `KG_HIGH_CONF_MIN_CANDIDATES = 3` kèm chú thích.
Việc cần làm là bổ sung điều kiện này vào phương trình (15) của bài báo.

Ngoài ra code dùng `>` còn bài dùng `≥`; đã sửa code thành `>=` cho khớp mặt chữ.
Với điểm số thực, khác biệt tại đúng mốc 0,85 gần như không xảy ra.

---

## 4. Tài sản — đã khôi phục đủ (cập nhật 13/08)

Quét `D:\Download` tìm được **toàn bộ** tài sản trước đó tưởng đã mất. Thư mục
`D:\Download\Medical_NLP\` là cây làm việc of-record đầy đủ.

| Tệp | Vai trò | Trạng thái |
|---|---|---|
| `medical_chunks_hierarchical.csv` | corpus RAG + từ điển triệu chứng | **có**, 8.214 đoạn, đã chép vào `data/` |
| `faiss_medical_local_phobert.index` | chỉ mục dense | **có**, IndexFlatIP 768 chiều, đã chép vào `data/` |
| `phobert-medical-constractive` | bộ mã hóa tương phản | có, hai bản trùng SHA-256 |
| `medical_data_cleaned.csv` | nguồn dựng đồ thị | có, đã chép vào `data/` |
| `test_300_paraphrased.csv` | tập kiểm thử 300 mẫu | có, đã chép vào `data/` |
| `ICD10_cleaned.csv` | danh mục 13.081 mã | có, đã chép vào `data/` |

Kiểm chứng toàn vẹn đã thực hiện:

- FAISS: `IxFI` (IndexFlatIP), `d = 768`, `ntotal = 8.214`; kích thước tệp bằng
  đúng `8214 × 768 × 4` cộng 45 byte header.
- CSV corpus: đúng 8.214 dòng → **khớp 1:1 theo dòng với chỉ mục**. Điều này bắt
  buộc vì `retrieve_rag()` tra ngược bằng `df_rag.iloc[idx]`.
- Cột cần thiết đều có: `disease_name`, `positive_symptoms`. Từ điển triệu chứng
  dựng được 6.955 mục.
- Ba bản chỉ mục FAISS (`Medical_NLP/`, `Medical_NLP_Hien/data/`,
  `paraphrase_aug/`) **trùng SHA-256**.
- Ba bản `medkgbert_dx.py` (hai trên Drive, một trong sao lưu) **trùng byte** —
  bản đã đưa vào repo là đúng bản of-record, không có mơ hồ.
- `medical_data_cleaned.csv` và `test_300_paraphrased.csv` trùng SHA-256 với bản
  trong gói `solieu_yte_new/`.

**Đính chính bản đối soát 12/08:** tài liệu đó liệt kê `rag_chunks.csv` và
`medical_chunks_hierarchical.csv` như hai tài sản riêng. Chúng trùng SHA-256
(`bbdc3952…`) — cùng một tệp mang hai tên ở hai nơi. Chỉ có một corpus.

Kết luận: **không còn gì chặn tái lập.** Điều kiện còn lại chỉ là môi trường chạy
(GPU cho BGE-reranker, một instance Neo4j đã nạp đồ thị v2).

---

## 5. Thay đổi đã thực hiện trong repo

Thêm mới:
- `code/rag/medkgbert_dx.py` — hệ thống of-record, cấu hình đã chỉnh theo Phụ lục D
- `code/database/import_neo4j_v2.py` — dựng schema 5 lớp / 5 quan hệ
- `code/database/embed_constractive.py` — nhúng 768 chiều vào `TrieuChung`
- `data/medical_data_cleaned.csv`, `data/test_300_paraphrased.csv`,
  `data/ICD10_cleaned.csv`, `data/medical_chunks_hierarchical.csv`,
  `data/faiss_medical_local_phobert.index`
- `data/ASSETS_MANIFEST.md` + `.sha256` — bản kê toàn vẹn mọi tài sản

Chỉnh trong `medkgbert_dx.py`:
- `USE_LLM_CROSS_RERANK` → `False` (khớp cấu hình báo cáo)
- τ_high đổi `>` thành `>=` (khớp phương trình 15)
- Tách hằng số có tên cho `RRF_K`, `RERANK_TOP_N`, `BGE_MAX_LENGTH`,
  `DENSE_TOPK_PER_VEC`, `BM25_TOP_N`, `KG_SYMP_MAX`, `KG_HIGH_CONF_MIN_CANDIDATES`
- Đường dẫn mô hình trỏ về bản sao lưu còn tồn tại
- Mật khẩu Neo4j đọc từ `.env` thay vì ghi cứng

Đánh dấu legacy (không xóa): `import_neo4j.py`, `PhoBert_neo4j.py`.
`import_neo4j.py` mở đầu bằng `MATCH (n) DETACH DELETE n` — chạy nhầm sẽ xóa sạch
đồ thị v2 rồi dựng lại schema cũ.

## 6. `.gitignore` — đã sửa

Quy tắc cũ `data/` chặn cả thư mục, cộng `*.json` chặn toàn bộ JSON. Đó chính là
cơ chế khiến dữ liệu nằm ngoài git và suýt mất trắng khi cài lại Windows.

Quy tắc mới bỏ hẳn lối chặn cả thư mục, chỉ loại trừ **nhị phân lớn**
(`*.index`, `*.faiss`, `*.pkl`, `*.zip`) — git không diff được nhị phân, mỗi lần
dựng lại chỉ mục là history phình thêm 25 MB.

Bốn tệp dẫn xuất được theo dõi: `medical_data_cleaned.csv`,
`test_300_paraphrased.csv`, `ICD10_cleaned.csv`, `medical_chunks_hierarchical.csv`.

### Điều kiện kèm theo: repo phải giữ private

Ngày 13/08 phát hiện repo đang **public** (`github.com/BaThanhHuynh/MedicalGraph`).
Đã kiểm tra lịch sử git: chỉ có 6 tệp từng được commit, toàn bộ là code thế hệ A —
không có `.env`, không có dữ liệu, không có khóa API nào bị lộ. Nghĩa là không có
gì phải khắc phục hồi tố.

Repo sau đó được chuyển sang private (xác nhận: API GitHub trả HTTP 404 khi truy
cập ẩn danh), và `medical_chunks_hierarchical.csv` mới được đưa vào git.

Tệp này chứa 24 MB văn bản y khoa **nguồn** từ tamanhhospital.vn và vinmec.com.
Mục 8.1 của bài báo nêu rõ không phát hành văn bản thô, và gói số liệu nộp kèm
cũng chỉ chứa dữ liệu đã trích xuất có cấu trúc. Vì vậy:

> **Trước khi chuyển repo sang public, phải gỡ tệp này khỏi toàn bộ lịch sử git**
> bằng `git filter-repo`. Xóa ở commit mới nhất là không đủ — nội dung vẫn nằm
> trong history và truy cập được.

Ràng buộc này được ghi lại ở cả `.gitignore` và `data/ASSETS_MANIFEST.md` để người
đọc sau này gặp nó trước khi kịp đổi visibility.

Mọi tệp bị loại trừ đều được ghi SHA-256 trong `data/ASSETS_MANIFEST.md`, kèm
`data/ASSETS_MANIFEST.sha256` để chạy `sha256sum -c`. Mất tệp nào sẽ phát hiện
được ngay thay vì phát hiện sau khi cài lại máy.

Dữ liệu thế hệ A (`data/json/`, `data/cleaned/`, `data/raw/`, `data/labeled/`) vẫn
để ngoài git nhưng giữ nguyên trên đĩa.

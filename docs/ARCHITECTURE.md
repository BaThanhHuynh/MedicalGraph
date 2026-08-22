# TÀI LIỆU KIẾN TRÚC HỆ THỐNG MEDKG-HRR (PRODUCTION ARCHITECTURE)

> **Dự án:** MedicalGraph — Đồ thị tri thức y tế neo ICD-10 và Khung truy xuất kết hợp MedKG-HRR.  
> **Phiên bản chuẩn:** v16 (Khớp 1:1 theo bài báo khoa học `Paper_Medical_PHT_v16.docx`).

---

## 1. TỔNG QUAN KIẾN TRÚC TOÀN HỆ THỐNG

MedKG-HRR là hệ thống phân tích triệu chứng y tế và xếp hạng phân biệt bệnh dựa trên 3 kênh truy xuất song song kết hợp mô hình tái xếp hạng nơ-ron và bộ điều khiển đồng thuận 3 nhánh:

```
[Bệnh nhân nhập mô tả triệu chứng tự do]
                    │
                    ▼
      ┌───────────────────────────┐
      │ (i) Trích xuất Triệu chứng│ (Dict matching + Cosine PhoBERT)
      └─────────────┬─────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌──────────────┐┌──────────────┐┌──────────────┐
│  Kênh Đồ thị ││  Kênh Dense  ││  Kênh Sparse │
│   Neo4j KG   ││  FAISS 768d  ││  BM25 Okapi  │
│  (4 tín hiệu)││ (Per-symptom)││  (Mở rộng)   │
└──────┬───────┘└──────┬───────┘└──────┬───────┘
       │               └───────┬───────┘
       │                       ▼
       │               ┌───────────────┐
       │               │ RRF Fusion    │ (Phương trình 10, k = 60)
       │               │ sRAG(d|q)     │
       │               └───────┬───────┘
       │                       │
       ▼                       ▼
   [sKG(d|q)]              [sRAG(d|q)]
       │                       │
       └───────────────┬───────┘
                       ▼
        ┌─────────────────────────────┐
        │  Hybrid Fusion v11          │
        │  (Bộ điều khiển 3 nhánh)    │
        ├─────────────────────────────┤
        │ 1. Consensus (Đồng thuận)   │ ──> [Top-K Bệnh lý ứng viên]
        │ 2. KG High Confidence       │ ──> [BGE Reranker v2-m3]
        │ 3. Disagree (KG-Anchored)   │ ──> [BGE Reranker v2-m3]
        └──────────────┬──────────────┘
                       ▼
        ┌─────────────────────────────┐
        │ Gắn Siêu Dữ Liệu ICD-10     │ (1.109 bệnh Thế hệ B)
        └──────────────┬──────────────┘
                       │
       [API Response / SSE Streaming]
                       │
                       ▼
        ┌─────────────────────────────┐
        │ Lớp Trợ Lý Gemini AI (UI)   │ (Tổng hợp & tư vấn bệnh nhân)
        └─────────────────────────────┘
```

---

## 2. CHI TIẾT CÁC THÀNH PHẦN (COMPONENTS)

### 2.1. Module Trích xuất Triệu chứng (Symptom Extraction)
* **Vị trí**: `backend/services/medkghrr_engine.py` (`extract_symptoms()`).
* **Cơ chế**:
  1. Tách câu văn thành các phân đoạn triệu chứng bằng regex và loại bỏ cụm từ mở đầu ("bác sĩ ơi", "dạo này", "tôi bị").
  2. Mã hóa vector các phân đoạn bằng **PhoBERT Contrastive** (`SegmentedEncoder` với `pyvi`).
  3. So khớp cosine similarity với từ điển **6.955 triệu chứng chuẩn** (được tiền lưu trong `data/symptom_embeddings_cache.npy`).
  4. Lọc bỏ các ứng viên dưới ngưỡng `DICT_SIM_THRESHOLD=0.55` và `DICT_MIN_TOKEN_OVR=1`.

### 2.2. Kênh Đồ Thị Tri Thức Neo4j (Kênh KG - Schema v2)
* **Vị trí**: `backend/services/medkghrr_engine.py` (`retrieve_kg()`).
* **Schema v2 (Thế hệ B)**:
  * **5 Lớp Node**: `BenhLy` (1.109 nodes), `TrieuChung` (7.558 nodes), `ICD10` (1.109 nodes), `ChuyenKhoa`, `XetNghiem`.
  * **5 Quan hệ**: `CO_TRIEU_CHUNG`, `LOAI_TRU_TRIEU_CHUNG`, `CO_MA_ICD10`, `THUOC_CHUYEN_KHOA`, `DUOC_CHAN_DOAN_BANG`.
* **4 Tín hiệu Cypher**:
  1. `Vector Index Search`: Tìm kiếm vector trên thuộc tính `TrieuChung.embedding` (768 chiều, index: `trieu_chung_vector_index`).
  2. `Bigram CONTAINS`: Bắt các cụm 2 từ có nghĩa trong tên triệu chứng.
  3. `Unigram CONTAINS`: Bắt các từ đơn có ý nghĩa lâm sàng ($\ge 5$ ký tự).
  4. `Symptom Full CONTAINS`: So khớp trực tiếp chuỗi triệu chứng.

### 2.3. Kênh Truy Xuất Dày (FAISS Dense Retrieval)
* **Vị trí**: `backend/services/medkghrr_engine.py` (`retrieve_rag()`).
* **Cơ chế**:
  * Chạy tìm kiếm láng giềng gần nhất (IndexFlatIP) trên **từng vector triệu chứng đã bóc tách** + **vector câu truy vấn gốc** trên 8.214 văn bản chunks phân tầng (`data/faiss_medical_local_phobert.index`).
  * Lấy `DENSE_TOPK_PER_VEC=5` ứng viên cho mỗi vector và tích lũy điểm tương đồng.

### 2.4. Kênh Truy Xuất Thưa (BM25 Okapi Sparse Retrieval)
* **Vị trí**: `backend/services/medkghrr_engine.py` (`retrieve_rag()`).
* **Cơ chế**:
  * Mở rộng query bằng danh sách triệu chứng bóc tách: `q_exp = query + " " + " ".join(symptoms)`.
  * Tra cứu trên 8.214 tài liệu tiền phân đoạn từ `data/bm25_tokenized_corpus.pkl` lấy top 50 văn bản có điểm BM25 cao nhất.

### 2.5. Hợp Nhất Xếp Hạng Nghịch Đảo (RRF Fusion)
* **Công thức**:
  $$s_{\text{RAG}}(d|q) = \sum_{c \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_c(d|q)}, \quad k = 60$$

### 2.6. Tái Xếp Hạng Nơ-ron (BGE Reranker)
* **Mô hình**: `BAAI/bge-reranker-v2-m3` (Cross-Encoder 560M params).
* **Đóng góp**: Tái xếp hạng ngữ cảnh sâu giữa câu truy vấn $q$ và văn bản mô tả bệnh học $d$, đóng góp cải thiện $+16,67\text{ pp}$ Hit@1 theo bài báo.

### 2.7. Bộ Điều Khiển Đồng Thuận 3 Nhánh (Hybrid Fusion v11)
1. **Nhánh 1: Consensus (Đồng thuận)**:
   * Kích hoạt khi: $\text{top-1}_{\text{RAG}} == \text{top-1}_{\text{KG}}$.
   * Kết hợp tuyến tính $s = 0.25 s_{\text{RAG}} + 0.75 s_{\text{KG}} + 0.40$ (Consensus Bonus).
   * Bỏ qua reranking vì hai kênh đã đồng thuận ở độ tin cậy cao.
2. **Nhánh 2: KG High Confidence**:
   * Kích hoạt khi: $C_{\text{KG}} \ge 0.85$ và số ứng viên KG $\ge 3$.
   * Tái xếp hạng top 10 KG bằng BGE Reranker: $s = 0.55 \cdot \sigma(\text{bge}) + 0.45 \cdot s_{\text{KG}} + 0.20 \cdot \mathbb{I}_{d \in \text{RAG}_5}$.
3. **Nhánh 3: Disagree (Neo Đồ Thị KG - KG-Anchored Fallback)**:
   * Kích hoạt khi hai kênh bất đồng.
   * Tập hợp pool ứng viên (Top 10 KG + Top 3 RAG), rerank với trọng số cân bằng:
     $s = 0.40 \cdot \sigma(\text{bge}) + 0.40 \cdot s_{\text{KG}} + 0.20 \cdot s_{\text{RAG}} + \text{Bonuses}$.

---

## 3. CẤU TRÚC THƯ MỤC CHUẨN PRODUCTION

```
MedicalGraph/
├── backend/                        # Web Backend Service (FastAPI)
│   ├── config.py                   # Cấu hình toàn hệ thống & tham số MedKG-HRR
│   ├── main.py                     # Entry point khởi chạy ứng dụng FastAPI
│   ├── routers/                    # API Endpoints
│   │   ├── api_chat.py             # Chat streaming SSE & xử lý tương tác
│   │   ├── api_config.py           # Quản lý cấu hình & trạng thái
│   │   └── api_search.py           # Endpoint tìm kiếm & /api/pipeline-info
│   └── services/                   # Logic nghiệp vụ lõi
│       ├── medkghrr_engine.py      # Bộ máy MedKG-HRR 3 kênh + 3 nhánh (Of-Record)
│       └── rag_engine.py           # Tích hợp Gemini LLM & RAG Presentation Layer
│
├── frontend/                       # Giao diện người dùng Web (Vanilla JS + CSS)
│   ├── index.html                  # Giao diện chính (Clinical Dashboard)
│   ├── style.css                   # Hệ thống Style (Dark mode, Glassmorphism)
│   └── script.js                   # Xử lý sự kiện, SSE Typewriter, Card Rendering
│
├── code/                           # Nghiên cứu & Pipeline Dữ liệu Khoa học
│   ├── database/                   # Script import đồ thị tri thức Neo4j
│   │   ├── import_neo4j_v2.py      # Import Schema v2 (Thế hệ B - 1.109 bệnh)
│   │   └── embed_constractive.py   # Nhúng vector PhoBERT 768d cho TrieuChung nodes
│   ├── rag/                        # Mã nguồn nghiên cứu of-record
│   │   └── medkgbert_dx.py         # Pipeline MedKG-HRR chuẩn bài báo v16
│   ├── extract/                    # Trích xuất thực thể và triệu chứng
│   ├── processdata/                # Tiền xử lý dữ liệu y tế
│   ├── evaluate/                   # Đánh giá độ chính xác, Fleiss' Kappa
│   └── thucnghiem_TN4_TN5/         # Notebooks & scripts tái lập thí nghiệm TN1-TN7
│
├── data/                           # Bộ tài nguyên dữ liệu đã kiểm định (Artifacts)
│   ├── ASSETS_MANIFEST.md          # Danh mục tài sản & mô tả kỹ thuật
│   ├── faiss_medical_local_phobert.index  # FAISS Index (8.214 vectors 768d)
│   ├── medical_chunks_hierarchical.csv    # 8.214 đoạn văn bản y tế phân tầng
│   ├── medical_data_cleaned.csv           # 1.109 bệnh lý chuẩn neo ICD-10
│   ├── bm25_tokenized_corpus.pkl          # BM25 Tokenized Corpus Cache
│   └── symptom_embeddings_cache.npy       # PhoBERT Embeddings Cache (6.955 triệu chứng)
│
├── models/                         # Trọng số mô hình cục bộ
│   └── phobert-medical-contrastive/       # Model PhoBERT contrastive đã fine-tune
│
├── docs/                           # Tài liệu kỹ thuật & số liệu chính thức
│   ├── ARCHITECTURE.md             # Tài liệu kiến trúc sản phẩm (file này)
│   ├── so_lieu_chot.json           # Bảng số liệu đã công bố của bài báo
│   └── ton_dong_bai_bao.md         # Ghi chú các điểm kiểm soát chất lượng
│
├── .env.example                    # File mẫu cấu hình môi trường
├── requirements.txt                # Danh sách thư viện phụ thuộc chuẩn
├── check_system.py                 # Script kiểm tra sức khỏe hệ thống 1-click
└── README.md                       # Hướng dẫn cài đặt & vận hành tổng thể
```

---

## 4. QUY TRÌNH TÁI LẬP HỆ THỐNG TỪ ĐẦU (REPRODUCIBILITY GUIDE)

### Bước 1: Chuẩn bị môi trường
```bash
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### Bước 2: Dựng cơ sở dữ liệu đồ thị Neo4j
1. Khởi động Neo4j (Local hoặc Aura).
2. Xóa dữ liệu cũ: `MATCH (n) DETACH DELETE n;`
3. Import đồ thị: `python code/database/import_neo4j_v2.py`
4. Số hóa vector triệu chứng: `python code/database/embed_constractive.py`
5. Tạo vector index trong Neo4j Browser:
   ```cypher
   CREATE VECTOR INDEX trieu_chung_vector_index IF NOT EXISTS
   FOR (t:TrieuChung) ON (t.embedding)
   OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};
   ```

### Bước 3: Kiểm tra toàn diện hệ thống
```bash
python check_system.py
```

### Bước 4: Vận hành Web Platform
```bash
uvicorn backend.main:app --reload --port 8000
```
Truy cập: `http://localhost:8000`.

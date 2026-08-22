# MedKG-HRR: Hệ Thống Đồ Thị Tri Thức Y Khoa & Khung Truy Xuất Kết Hợp (Neo ICD-10)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.15+-008CC1.svg)](https://neo4j.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Dense%20Search-yellow.svg)](https://github.com/facebookresearch/faiss)
[![PhoBERT](https://img.shields.io/badge/PhoBERT-Contrastive%20768d-orange.svg)](https://github.com/VinAIResearch/PhoBERT)

**MedKG-HRR** (*Medical Knowledge Graph - Hybrid Retrieval & Reranking*) là nền tảng phân tích triệu chứng y tế và gợi ý chẩn đoán phân biệt tiếng Việt, neo chuẩn danh mục quốc tế **ICD-10** và cấu trúc hóa bằng đồ thị tri thức **Neo4j (Schema v2)**.

Dự án được xây dựng và chuẩn hóa khớp 1:1 theo công bố khoa học trong bài báo **`Paper_Medical_PHT_v16.docx`**.

---

## ✨ Điểm Nổi Bật Của Kiến Trúc MedKG-HRR

1. **Trích xuất Triệu chứng Tự động**: Bóc tách triệu chứng từ câu văn tự do của bệnh nhân bằng thuật toán Dict Matching kết hợp vector hóa ngữ nghĩa PhoBERT.
2. **Truy xuất 3 Kênh Song Song (Multi-channel Hybrid Retrieval)**:
   * 🌲 **Kênh Đồ thị (Neo4j KG)**: Truy vấn Cypher 4 tín hiệu (Vector index, Bigram, Unigram, Symptom CONTAINS) trên mạng lưới 1.109 bệnh và 7.558 triệu chứng.
   * ⚡ **Kênh Dense (FAISS IndexFlatIP)**: Tìm kiếm vector ngữ nghĩa 768 chiều theo từng triệu chứng độc lập trên 8.214 văn bản y khoa phân tầng.
   * 🔍 **Kênh Sparse (BM25 Okapi)**: Khai thác từ khóa mở rộng trên kho dữ liệu phân đoạn y tế.
3. **Hợp nhất Xếp hạng Nghịch đảo (RRF $k=60$)**: Kết hợp kênh Dense và Sparse theo chuẩn phương trình (10).
4. **Tái xếp hạng Nơ-ron (BGE-Reranker-v2-m3)**: Tinh chỉnh thứ tự ứng viên qua mô hình Cross-Encoder chuyên sâu (+16,67 pp Hit@1).
5. **Bộ điều khiển Đồng thuận 3 nhánh (Hybrid Fusion v11)**:
   * **Consensus**: Đồng thuận cao khi Top-1 KG trùng Top-1 RAG.
   * **KG High Confidence**: Neo vào đồ thị khi độ tin cậy KG vượt trội ($C_{\text{KG}} \ge 0.85$).
   * **Disagree (KG-Anchored)**: Cân bằng trọng số và tái xếp hạng khi hai kênh bất đồng.
6. **Lớp Trợ lý Gemini AI (UI/UX Layer)**: Giải thích và phân tích lâm sàng tương tác dạng Server-Sent Events (SSE) theo thời gian thực.

---

## 📁 Cấu Trúc Thư Mục Chuẩn Production

Chi tiết sơ đồ khối và logic từng thành phần xem tại: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```text
MedicalGraph/
├── backend/                  # FastAPI Backend API
│   ├── config.py             # Cấu hình toàn hệ thống & tham số MedKG-HRR
│   ├── main.py               # Entry point ASGI Server
│   ├── routers/              # API endpoints (/api/chat, /api/search, /api/config)
│   └── services/             # Core engines (medkghrr_engine.py, rag_engine.py)
├── frontend/                 # Web Clinical Dashboard (HTML/CSS/JS thuần, Glassmorphism)
├── code/                     # Pipeline Dữ liệu, Đồ thị & Thí nghiệm Khoa học
│   ├── database/             # Import Neo4j v2 & Vector Embeddings
│   ├── rag/                  # Mã nguồn nghiên cứu of-record (medkgbert_dx.py)
│   ├── extract/              # Trích xuất triệu chứng và thực thể
│   ├── evaluate/             # Đánh giá độ chính xác, Fleiss' Kappa
│   └── thucnghiem_TN4_TN5/   # Notebooks tái lập các thí nghiệm TN1–TN7
├── data/                     # Dữ liệu chuẩn đã kiểm định (FAISS, CSV, PKL, Cache)
├── models/                   # Mô hình PhoBERT contrastive cục bộ
├── docs/                     # Tài liệu kiến trúc & Bảng số liệu chốt
├── .env.example              # Mẫu cấu hình môi trường
├── requirements.txt          # Danh sách dependencies đã phân nhóm
├── check_system.py           # Script kiểm tra sức khỏe hệ thống 1-click
└── README.md                 # Hướng dẫn sử dụng tổng thể (file này)
```

---

## 🚀 Hướng Dẫn Cài Đặt & Vận Hành (Quickstart)

### Bước 1: Chuẩn bị môi trường Python

Khuyến nghị sử dụng **Python 3.10 hoặc 3.11**:

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Trên Windows PowerShell:
.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

# Cài đặt toàn bộ thư viện phụ thuộc
python -m pip install -r requirements.txt
```

### Bước 2: Thiết lập file môi trường `.env`

Sao chép từ file mẫu `.env.example`:
```powershell
cp .env.example .env
```
Mở file `.env` và điền:
* `GEMINI_API_KEY`: API Key lấy miễn phí tại [Google AI Studio](https://aistudio.google.com/app/apikey).
* `NEO4J_PASSWORD`: Mật khẩu cơ sở dữ liệu Neo4j của bạn.

---

### Bước 3: Nạp dữ liệu Đồ thị Tri thức vào Neo4j (Thế hệ B)

1. Mở **Neo4j Desktop** hoặc dịch vụ Neo4j cục bộ và bấm **Start**.
2. Xóa dữ liệu cũ trong Neo4j Browser (`http://localhost:7474`):
   ```cypher
   MATCH (n) DETACH DELETE n;
   ```
3. Import đồ thị 1.109 bệnh lý chuẩn:
   ```powershell
   python code/database/import_neo4j_v2.py
   ```
4. Số hóa vector triệu chứng 768 chiều:
   ```powershell
   python code/database/embed_constractive.py
   ```
5. Tạo Vector Index trong Neo4j Browser:
   ```cypher
   CREATE VECTOR INDEX trieu_chung_vector_index IF NOT EXISTS
   FOR (t:TrieuChung) ON (t.embedding)
   OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};
   ```

---

### Bước 4: Kiểm tra sức khỏe hệ thống (Health Check)

Chạy script chẩn đoán tự động:
```powershell
python check_system.py
```
Script sẽ kiểm tra toàn bộ môi trường Python, CUDA/CPU, các tệp dữ liệu, mô hình PhoBERT, kết nối Neo4j và API Key.

---

### Bước 5: Khởi chạy Web Platform

```powershell
uvicorn backend.main:app --reload --port 8000
```

Mở trình duyệt tại: **`http://localhost:8000`**

---

## 📡 API Endpoints Chính

| Phương thức | Đường dẫn | Chức năng |
|---|---|---|
| `POST` | `/api/chat/stream` | Tư vấn y tế tương tác thời gian thực (SSE Streaming) |
| `POST` | `/api/search` | Tra cứu Top-K bệnh lý qua MedKG-HRR Pipeline |
| `GET` | `/api/pipeline-info` | Kiểm tra trạng thái các components của MedKG-HRR |
| `GET` | `/api/config` | Lấy cấu hình runtime hiện tại |
| `POST` | `/api/config` | Cập nhật runtime (API Key, Model, Top-K, Nhiệt độ) |

---

## 📊 Tái Lập Thí Nghiệm & Đánh Giá

Tất cả số liệu công bố trong bài báo được lưu trữ có kiểm tra mã băm SHA-256 tại [`docs/so_lieu_chot.json`](docs/so_lieu_chot.json).

* **Thực nghiệm TN1, TN2, TN3**: Xem tại thư mục `code/evaluate/`.
* **Thực nghiệm TN4, TN5, TN7 (Prompt Sensitivity & CoT)**: Xem các Jupyter Notebooks trong `code/thucnghiem_TN4_TN5/`.

---

## 🛡️ Giấy Phép & Bảo Mật

* **Bảo mật**: File `.env` chứa thông tin nhạy cảm đã được cấu hình trong `.gitignore`. Tuyệt đối không commit API Key hoặc mật khẩu lên repository.
* **Tuyên bố miễn trừ trách nhiệm y tế**: Kết quả phân tích từ AI chỉ mang tính chất tham khảo học thuật và định hướng thông tin y tế theo hệ thống MedKG-HRR, không thay thế cho chẩn đoán hoặc chỉ định trực tiếp từ Bác sĩ chuyên khoa.
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv

# Nạp file .env từ thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class SystemConfig:
    """Quản lý cấu hình tập trung cho toàn bộ hệ thống MedicalGraph Backend.
    
    Tham số MedKG-HRR pipeline khớp 1:1 với bài báo v16 (Mục 4, Phụ lục D)
    và implementation of-record trong code/rag/medkgbert_dx.py.
    """
    
    def __init__(self):
        self.BASE_DIR: Path = BASE_DIR
        
        # ============================================================
        # ĐƯỜNG DẪN MODEL & DỮ LIỆU
        # ============================================================
        
        # PhoBERT-base tinh chỉnh tương phản (Mục 5.2 bài báo)
        default_phobert = BASE_DIR / "models" / "phobert-medical-contrastive"
        self.PHOBERT_MODEL_PATH: str = os.getenv("PHOBERT_MODEL_PATH", str(default_phobert))
        self.MODEL_PHOBERT_DIR: str = self.PHOBERT_MODEL_PATH
        
        # FAISS IndexFlatIP 768 chiều, 8.214 vectors (Mục 4)
        self.FAISS_INDEX_PATH: str = os.getenv(
            "FAISS_INDEX_PATH", 
            str(BASE_DIR / "data" / "faiss_medical_local_phobert.index")
        )
        # Corpus RAG: 8.214 chunks phân tầng (Mục 4)
        self.CSV_RAG_CHUNKS: str = os.getenv(
            "CSV_RAG_CHUNKS", 
            str(BASE_DIR / "data" / "medical_chunks_hierarchical.csv")
        )
        self.CSV_CHUNKS: str = self.CSV_RAG_CHUNKS
        # Database 1.109 bệnh lý chuẩn thế hệ B, neo ICD-10 (Mục 3)
        self.CSV_MEDICAL: str = os.getenv(
            "CSV_MEDICAL", 
            str(BASE_DIR / "data" / "medical_data_cleaned.csv")
        )
        
        # BGE-reranker-v2-m3 (Mục 4, phương trình 13-14; Phụ lục D)
        self.BGE_RERANKER_PATH: str = os.getenv(
            "BGE_RERANKER_PATH", "BAAI/bge-reranker-v2-m3"
        )
        
        # ============================================================
        # CẤU HÌNH GEMINI LLM (phần UI bổ trợ, KHÔNG thuộc pipeline MedKG-HRR)
        # Bài báo nêu rõ: hệ thống "không sinh văn bản chẩn đoán"
        # ============================================================
        self.GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        self.GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
        
        # ============================================================
        # CẤU HÌNH NEO4J — Kênh đồ thị KG (Mục 4, phương trình 8-9)
        # Schema v2: 5 lớp (BenhLy, TrieuChung, ICD10, ChuyenKhoa, XetNghiem)
        #            5 quan hệ (CO_TRIEU_CHUNG, LOAI_TRU_TRIEU_CHUNG, CO_MA_ICD10,
        #                       THUOC_CHUYEN_KHOA, DUOC_CHAN_DOAN_BANG)
        # ============================================================
        self.NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
        self.NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
        
        # ============================================================
        # THAM SỐ PIPELINE MEDKG-HRR (Phụ lục D, medkgbert_dx.py Config)
        # ============================================================
        
        # --- Retrieval (Phụ lục D mục I) ---
        self.TOP_K: int = int(os.getenv("TOP_K", "10"))  # Bài báo: TOP_K = 10
        self.RRF_K: int = 60                  # Phương trình (10): k = 60
        self.RERANK_TOP_N: int = 5            # Số ứng viên đưa vào BGE reranker
        self.BGE_MAX_LENGTH: int = 1024       # Rerank Max Window Length
        self.DENSE_TOPK_PER_VEC: int = 5      # Số láng giềng FAISS mỗi vector triệu chứng
        self.BM25_TOP_N: int = 50             # Độ sâu BM25 trước RRF
        self.KG_SYMP_MAX: int = 6             # Số triệu chứng/bệnh khi dựng subgraph
        
        # --- Trích xuất triệu chứng (Mục 4, bước (i)) ---
        self.DICT_SIM_THRESHOLD: float = 0.55  # Ngưỡng cosine dict matching
        self.DICT_MIN_TOKEN_OVR: int = 1       # Overlap token tối thiểu
        self.DICT_MIN_SYM_LEN: int = 3
        self.DICT_MAX_SYM_LEN: int = 80
        self.EXTRACT_MIN_DICT_COUNT: int = 2   # Số triệu chứng tối thiểu từ dict
        
        # --- KG Vector Index ---
        self.KG_VECTOR_INDEX: str = "trieu_chung_vector_index"
        self.KG_VECTOR_LIMIT: int = 8
        self.KG_VECTOR_SCORE_MIN: float = 0.45
        
        # --- Hybrid Fusion v11 (Phụ lục D mục II, phương trình 11/13/15/20) ---
        self.ALPHA_CONSENSUS: float = 0.25       # eq (11): 0.25 sRAG + 0.75 sKG
        self.CONSENSUS_BONUS: float = 0.40       # eq (11): Consensus Bonus
        self.KG_HIGH_CONF_THRESHOLD: float = 0.85  # τ_high, eq (15)
        self.KG_MED_CONF_THRESHOLD: float = 0.65   # τ_med, eq (20)
        # Ràng buộc of-record: KG High Conf chỉ kích hoạt khi ≥ 3 ứng viên
        self.KG_HIGH_CONF_MIN_CANDIDATES: int = 3
        
        # --- Component Toggles (graceful fallback) ---
        self.USE_BGE_RERANKER: bool = os.getenv("USE_BGE_RERANKER", "true").lower() == "true"
        self.USE_KG_RETRIEVAL: bool = os.getenv("USE_KG_RETRIEVAL", "true").lower() == "true"
        self.USE_SYMPTOM_EXTRACTION: bool = True  # Luôn bật — thiếu bước này là sai kiến trúc
        
    def update(self, **kwargs):
        """Cập nhật cấu hình runtime từ API."""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
                # Đồng bộ lại biến môi trường nếu cần
                if key == "GEMINI_API_KEY":
                    os.environ["GEMINI_API_KEY"] = str(value)
                    os.environ["GOOGLE_API_KEY"] = str(value)
                    
    def to_dict(self) -> dict:
        """Xuất thông tin cấu hình an toàn (ẩn API key một phần)."""
        masked_key = ""
        if self.GEMINI_API_KEY:
            if len(self.GEMINI_API_KEY) > 8:
                masked_key = f"{self.GEMINI_API_KEY[:4]}...{self.GEMINI_API_KEY[-4:]}"
            else:
                masked_key = "***"
                
        return {
            "gemini_api_key_configured": bool(self.GEMINI_API_KEY),
            "gemini_api_key_masked": masked_key,
            "gemini_model": self.GEMINI_MODEL,
            "temperature": self.TEMPERATURE,
            "top_k": self.TOP_K,
            "phobert_model_path": self.PHOBERT_MODEL_PATH,
            "faiss_index_path": self.FAISS_INDEX_PATH,
            "csv_rag_chunks": self.CSV_RAG_CHUNKS,
            "csv_medical": self.CSV_MEDICAL,
            "neo4j_uri": self.NEO4J_URI,
            "bge_reranker_path": self.BGE_RERANKER_PATH,
            "pipeline_config": {
                "rrf_k": self.RRF_K,
                "rerank_top_n": self.RERANK_TOP_N,
                "dense_topk_per_vec": self.DENSE_TOPK_PER_VEC,
                "bm25_top_n": self.BM25_TOP_N,
                "alpha_consensus": self.ALPHA_CONSENSUS,
                "kg_high_conf_threshold": self.KG_HIGH_CONF_THRESHOLD,
            },
            "components_enabled": {
                "symptom_extraction": self.USE_SYMPTOM_EXTRACTION,
                "kg_retrieval": self.USE_KG_RETRIEVAL,
                "bge_reranker": self.USE_BGE_RERANKER,
            },
            "available_models": [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-3.1-flash-lite"
            ]
        }

# Global singleton config
config = SystemConfig()

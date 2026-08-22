"""
MedKG-HRR Engine cho Web Backend
=================================
Port từ code/rag/medkgbert_dx.py (implementation of-record, bài báo v16)
với kiến trúc khớp 1:1 bài báo Mục 4:

  - Trích xuất triệu chứng: Dict matching (pyvi + PhoBERT cosine)
  - Kênh đồ thị KG: Cypher trên Neo4j (4 tín hiệu: vector, bigram, unigram, symptom)
  - Kênh RAG: FAISS dense (per-symptom) + BM25 sparse → RRF k=60
  - BGE Reranker: bge-reranker-v2-m3 (cross-encoder)
  - Hybrid Fusion v11: 3 nhánh (consensus / kg_high_conf / disagree_kg_anchored)

Mỗi component lazy-load và graceful fallback khi thiếu tài nguyên.
"""
from __future__ import annotations

import os
import re
import time
import logging
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from backend.config import config

logger = logging.getLogger("medkghrr_engine")

# ============================================================
#  STOPWORDS / REGEX (giữ nguyên từ medkgbert_dx.py)
# ============================================================
_STOP_DICT = {
    'và', 'hoặc', 'của', 'là', 'có', 'bị', 'rất', 'nhiều', 'ít', 'này', 'đó',
    'như', 'nên', 'phải', 'tôi', 'bạn', 'các', 'một', 'hai', 'khi', 'lúc',
    'cảm', 'thấy', 'hay', 'thường', 'dạo', 'gần', 'đây', 'mấy', 'ngày', 'hôm'
}
_KG_STOP = {
    'cảm', 'thấy', 'bị', 'và', 'hoặc', 'rất', 'nhiều', 'ít',
    'tôi', 'của', 'là', 'có', 'được', 'mất', 'đã', 'sẽ',
    'này', 'đó', 'như', 'nên', 'phải', 'cũng', 'còn',
    'hay', 'thường', 'đôi', 'khi', 'lúc', 'một', 'hai'
}
_KG_GENERIC = {'đau', 'sốt', 'ho', 'mệt', 'nôn', 'ngứa', 'sưng', 'rát'}

_SEP_RE = re.compile(
    r'\s*(?:,|\.|;|\bvà\b|\bhoặc\b|\bkèm(?:\s+theo)?\b|'
    r'\bthỉnh thoảng\b|\bcòn\b|\bngoài ra\b|\bđôi khi\b|\bcùng\b)\s*',
    re.IGNORECASE)

_PREAM_STRIP = re.compile(
    r'^(bác sĩ ơi|tôi muốn hỏi|dạo (này|gần đây)|mấy (hôm|ngày) nay'
    r'|tôi (bị|cảm thấy|hay|thường|gặp tình trạng)'
    r'|người tôi|gần đây tôi|tôi cảm thấy|tư vấn giúp tôi'
    r'|(bị|cảm thấy|hay|thường) )',
    re.IGNORECASE)


def _toks(s):
    return {w for w in re.findall(r'\w+', s.lower())
            if len(w) >= 3 and w not in _STOP_DICT}


def _get_bigrams(s):
    words = [w for w in re.findall(r'\w+', s.lower()) if w not in _KG_STOP]
    return [f"{words[i]} {words[i+1]}"
            for i in range(len(words) - 1)
            if len(f"{words[i]} {words[i+1]}") >= 6]


def _get_significant_unigrams(s):
    return [w for w in re.findall(r'\w+', s.lower())
            if len(w) >= 5 and w not in _KG_STOP and w not in _KG_GENERIC]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


# ============================================================
#  SEGMENTED ENCODER (giữ nguyên từ medkgbert_dx.py)
# ============================================================
class SegmentedEncoder:
    """Auto pyvi.ViTokenizer.tokenize trước khi encode."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        segmented = [self.tokenizer.tokenize(t) for t in texts]
        out = self.model.encode(segmented, **kwargs)
        return out[0] if single else out

    def __getattr__(self, name):
        return getattr(self.model, name)


# ============================================================
#  MEDKG-HRR ENGINE (chuẩn bài báo v16)
# ============================================================
class MedKGHRREngine:
    """
    MedKG-HRR Pipeline chuẩn bài báo v16, port từ medkgbert_dx.py.
    
    Kiến trúc 3 kênh + 3 nhánh:
    - Kênh KG: Neo4j Cypher (vector index + bigram + unigram + symptom CONTAINS)
    - Kênh Dense: FAISS PhoBERT contrastive (per-symptom + query)
    - Kênh Sparse: BM25 Okapi (query mở rộng bằng triệu chứng)
    - RRF Fusion k=60 cho Dense + Sparse
    - BGE-reranker-v2-m3 tái xếp hạng
    - Hybrid Fusion v11: Consensus / KG High Conf / Disagree (KG-anchored)
    """

    def __init__(self):
        # Lazy-load state
        self._device = None
        self._embed_model = None
        self._raw_embed = None
        self._bge_tokenizer = None
        self._bge_model = None
        self._neo4j_driver = None
        self._faiss_index = None
        self._df_rag = None
        self._corpus = None
        self._bm25 = None
        self._disease_to_text: Dict[str, str] = {}
        self._symptom_list: List[str] = []
        self._symptom_emb: Optional[np.ndarray] = None
        self._symptom_tokens: List[set] = []
        self._df_medical = None
        self._disease_dict: Dict[str, Dict] = {}

        # Runtime state
        self.is_ready = False
        self._components_status: Dict[str, str] = {}

    # ---------- device ----------
    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
            if self._device == "cpu" and config.USE_BGE_RERANKER:
                logger.warning("CUDA không khả dụng — BGE reranker sẽ chạy chậm trên CPU.")
        return self._device

    # ============================================================
    #  LAZY LOADERS
    # ============================================================
    def _load_embed(self):
        """Nạp PhoBERT contrastive encoder + pyvi SegmentedEncoder."""
        if self._embed_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        from pyvi import ViTokenizer
        phobert_dir = config.PHOBERT_MODEL_PATH
        if not os.path.exists(phobert_dir):
            raise FileNotFoundError(f"Không tìm thấy PhoBERT tại: {phobert_dir}")
        logger.info(f"[load] PhoBERT contrastive: {phobert_dir}")
        self._raw_embed = SentenceTransformer(phobert_dir, device=self.device)
        self._embed_model = SegmentedEncoder(self._raw_embed, ViTokenizer)
        self._components_status["phobert_encoder"] = "active"
        logger.info("[✓] PhoBERT contrastive đã sẵn sàng.")

    def _load_bge(self):
        """Nạp BGE-reranker-v2-m3 cross-encoder (lazy-load khi cần)."""
        if self._bge_model is not None or not config.USE_BGE_RERANKER:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            path = config.BGE_RERANKER_PATH
            logger.info(f"[load] Đang nạp BGE reranker: {path}...")
            self._bge_tokenizer = AutoTokenizer.from_pretrained(path)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            # Dùng torch_dtype hoặc dtype an toàn
            try:
                self._bge_model = AutoModelForSequenceClassification.from_pretrained(
                    path, dtype=dtype
                ).to(self.device)
            except TypeError:
                self._bge_model = AutoModelForSequenceClassification.from_pretrained(
                    path, torch_dtype=dtype
                ).to(self.device)
            self._bge_model.eval()
            self._components_status["bge_reranker"] = "active"
            logger.info("[✓] BGE reranker đã sẵn sàng.")
        except Exception as e:
            logger.warning(f"Không load được BGE ({e}) — tắt BGE rerank.")
            config.USE_BGE_RERANKER = False
            self._components_status["bge_reranker"] = f"disabled: {e}"

    def _load_neo4j(self):
        """Kết nối Neo4j cho kênh đồ thị KG."""
        if self._neo4j_driver is not None or not config.USE_KG_RETRIEVAL:
            return
        try:
            from neo4j import GraphDatabase
            self._neo4j_driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            # Test connection
            with self._neo4j_driver.session() as session:
                result = session.run("MATCH (b:BenhLy) RETURN count(b) AS cnt")
                cnt = result.single()["cnt"]
            self._components_status["neo4j_kg"] = f"active ({cnt} BenhLy nodes)"
            logger.info(f"[✓] Neo4j KG kết nối thành công: {cnt} nút BenhLy.")
        except Exception as e:
            logger.warning(f"Không kết nối được Neo4j ({e}) — tắt kênh KG.")
            config.USE_KG_RETRIEVAL = False
            self._neo4j_driver = None
            self._components_status["neo4j_kg"] = f"disabled: {e}"

    def _load_faiss_corpus(self):
        """Nạp FAISS index + corpus + BM25."""
        if self._bm25 is not None:
            return
        from pyvi import ViTokenizer
        import faiss

        # FAISS index
        faiss_path = config.FAISS_INDEX_PATH
        if os.path.exists(faiss_path):
            logger.info(f"[load] FAISS index: {faiss_path}")
            self._faiss_index = faiss.read_index(faiss_path)
            self._components_status["faiss_index"] = f"active ({self._faiss_index.ntotal} vectors)"
        else:
            logger.warning(f"Không tìm thấy FAISS index: {faiss_path}")
            self._faiss_index = None
            self._components_status["faiss_index"] = "missing"

        # RAG corpus
        chunks_path = config.CSV_RAG_CHUNKS
        if not os.path.exists(chunks_path):
            chunks_path = config.CSV_MEDICAL
            logger.warning(f"Không thấy CSV chunks — fallback {chunks_path}")

        try:
            self._df_rag = pd.read_csv(chunks_path, on_bad_lines='skip',
                                        engine='python', encoding='utf-8')
            if self._df_rag.shape[1] == 1:
                self._df_rag = pd.read_csv(chunks_path, delimiter=';',
                                            on_bad_lines='skip', engine='python',
                                            encoding='utf-8')
        except Exception:
            self._df_rag = pd.read_csv(chunks_path, delimiter=';',
                                        on_bad_lines='skip', engine='python',
                                        encoding='utf-8')

        self._corpus = self._df_rag.fillna('').astype(str).agg(' '.join, axis=1).tolist()

        # BM25 - Ưu tiên nạp từ data/bm25_tokenized_corpus.pkl nếu có để khởi động tức thì
        from rank_bm25 import BM25Okapi
        import pickle
        bm25_pkl = config.BASE_DIR / "data" / "bm25_tokenized_corpus.pkl"
        if os.path.exists(bm25_pkl):
            try:
                with open(bm25_pkl, "rb") as f:
                    tokenized = pickle.load(f)
                logger.info(f"[load] [✓] Đã nạp {len(tokenized)} tài liệu tiền phân đoạn từ {bm25_pkl}")
            except Exception as e:
                logger.warning(f"Lỗi đọc {bm25_pkl}: {e}, tiến hành phân đoạn mới...")
                tokenized = [ViTokenizer.tokenize(doc.lower()).split() for doc in self._corpus]
        else:
            tokenized = [ViTokenizer.tokenize(doc.lower()).split() for doc in self._corpus]

        self._bm25 = BM25Okapi(tokenized)
        self._components_status["bm25"] = f"active ({len(self._corpus)} docs)"
        logger.info(f"[✓] BM25 đã sẵn sàng cho {len(self._corpus)} văn bản y khoa.")

        # disease_to_text mapping
        name_col = "disease_name" if "disease_name" in self._df_rag.columns \
            else ("ten_benh" if "ten_benh" in self._df_rag.columns else None)
        if name_col:
            for idx, row in self._df_rag.iterrows():
                d = str(row[name_col]).strip().lower()
                if d and d != 'nan':
                    self._disease_to_text[d] = self._disease_to_text.get(d, "") + " " + self._corpus[idx]

    def _load_medical_database(self):
        """Nạp danh mục 1.109 bệnh chuẩn thế hệ B, neo ICD-10."""
        if self._df_medical is not None:
            return
        med_path = config.CSV_MEDICAL
        if not os.path.exists(med_path):
            logger.warning(f"Không tìm thấy file bệnh chuẩn: {med_path}")
            return

        try:
            self._df_medical = pd.read_csv(med_path, delimiter=";",
                                            on_bad_lines="skip", engine="python",
                                            encoding="utf-8")
        except Exception as e:
            logger.warning(f"Lỗi đọc {med_path}: {e}")
            return

        for _, row in self._df_medical.iterrows():
            disease_name = str(row.get("disease_name", row.get("ten_benh", ""))).strip()
            if not disease_name or disease_name.lower() == "nan":
                continue
            icd10 = str(row.get("icd10_code", row.get("ma_icd10", ""))).strip()
            if icd10.lower() == "nan":
                icd10 = ""
            icd_name = str(row.get("icd10_name", "")).strip()
            if icd_name.lower() == "nan":
                icd_name = ""
            specialty = str(row.get("chuyen_khoa_goc", row.get("chuyen_khoa", "Đa khoa / Nội khoa"))).strip()
            if specialty.lower() == "nan":
                specialty = "Đa khoa / Nội khoa"
            symptoms = str(row.get("positive_symptoms", row.get("positive_symptoms_norm", ""))).strip()
            if symptoms.lower() == "nan":
                symptoms = ""
            chan_doan = str(row.get("chan_doan_text", "")).strip()
            if chan_doan.lower() == "nan":
                chan_doan = ""

            self._disease_dict[disease_name.lower()] = {
                "name": disease_name,
                "icd10": icd10,
                "icd10_name": icd_name,
                "specialty": specialty,
                "symptoms": symptoms,
                "chan_doan": chan_doan
            }
        self._components_status["medical_db"] = f"active ({len(self._disease_dict)} diseases)"
        logger.info(f"[✓] Đã lập chỉ mục {len(self._disease_dict)} bệnh lý chuẩn ICD-10.")

    # ============================================================
    #  KHỞI TẠO (gọi 1 lần khi startup)
    # ============================================================
    def initialize(self):
        """Khởi tạo toàn bộ components MedKG-HRR."""
        try:
            logger.info("=" * 60)
            logger.info("  KHỞI TẠO MEDKG-HRR ENGINE (CHUẨN BÀI BÁO V16)")
            logger.info("=" * 60)

            # Core components (bắt buộc)
            self._load_embed()
            self._load_faiss_corpus()
            self._load_medical_database()

            # Symptom dictionary (bắt buộc theo bài báo)
            self.build_symptom_dictionary()

            # Optional components (graceful fallback & lazy-load)
            self._load_neo4j()
            if config.USE_BGE_RERANKER:
                self._components_status["bge_reranker"] = "enabled (lazy-load on demand)"

            self.is_ready = True
            logger.info("=" * 60)
            logger.info("  [✓] MEDKG-HRR ENGINE ĐÃ SẴN SÀNG")
            for comp, status in self._components_status.items():
                logger.info(f"    {comp}: {status}")
            logger.info("=" * 60)
        except Exception as e:
            self.is_ready = False
            logger.error(f"[X] Lỗi khởi tạo MedKG-HRR: {e}", exc_info=True)

    # ============================================================
    #  TRÍCH XUẤT TRIỆU CHỨNG (Mục 4 bước (i), medkgbert_dx.py v10)
    # ============================================================
    def build_symptom_dictionary(self, batch_size: int = 128):
        """Build từ điển triệu chứng từ corpus CSV kèm disk cache để khởi động tức thì."""
        if self._symptom_emb is not None:
            return
        self._load_embed()
        self._load_faiss_corpus()
        df = self._df_rag
        all_symps = set()
        for col in ['positive_symptoms_norm', 'positive_symptoms']:
            if col not in df.columns:
                continue
            for v in df[col].dropna():
                for s in str(v).split('|'):
                    s = s.strip().lower()
                    if config.DICT_MIN_SYM_LEN <= len(s) <= config.DICT_MAX_SYM_LEN:
                        all_symps.add(s)
        self._symptom_list = sorted(all_symps)
        if not self._symptom_list:
            logger.warning("Không build được symptom dictionary — CSV thiếu cột triệu chứng.")
            self._components_status["symptom_dict"] = "empty"
            return
        logger.info(f"[dict] Symptom dictionary: {len(self._symptom_list)} unique")
        
        # Disk cache để tránh encode lại 6.955 triệu chứng tại mỗi lần khởi động
        cache_path = os.path.join(str(config.BASE_DIR / "data"), "symptom_embeddings_cache.npy")
        if os.path.exists(cache_path):
            try:
                self._symptom_emb = np.load(cache_path)
                logger.info(f"[dict] [✓] Đã nạp cache embedding triệu chứng ({self._symptom_emb.shape}) từ {cache_path}")
            except Exception as e:
                logger.warning(f"Lỗi đọc cache {cache_path}: {e}, tiến hành encode mới...")
                self._symptom_emb = None

        if self._symptom_emb is None:
            logger.info(f"[dict] Đang mã hóa {len(self._symptom_list)} triệu chứng bằng PhoBERT (lần đầu)...")
            self._symptom_emb = self._embed_model.encode(
                self._symptom_list, batch_size=batch_size,
                show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
            )
            try:
                np.save(cache_path, self._symptom_emb)
                logger.info(f"[dict] [✓] Đã lưu cache embedding triệu chứng vào {cache_path}")
            except Exception as e:
                logger.warning(f"Không thể lưu cache: {e}")

        self._symptom_tokens = [_toks(s) for s in self._symptom_list]
        self._components_status["symptom_dict"] = f"active ({len(self._symptom_list)} symptoms)"

    def extract_symptoms(self, query: str, top_k_per_chunk: int = 3,
                         max_total: int = 8) -> List[str]:
        """
        Trích xuất triệu chứng từ query bằng Dict matching (Mục 4 bước (i)).
        Port nguyên vẹn từ medkgbert_dx.py extract_symptoms_dict().
        """
        if not config.USE_SYMPTOM_EXTRACTION or self._symptom_emb is None:
            return []

        chunks = [c.strip() for c in _SEP_RE.split(query)
                  if c.strip() and len(c.strip()) >= 3]
        cleaned = []
        for c in chunks:
            c2 = _PREAM_STRIP.sub('', c).strip()
            if len(c2) >= 3:
                cleaned.append(c2)
        if not cleaned:
            cleaned = chunks
        if not cleaned:
            return []

        query_tokens = _toks(query)
        chunk_emb = self._embed_model.encode(
            cleaned, batch_size=16, convert_to_numpy=True, normalize_embeddings=True
        )
        if chunk_emb.ndim == 1:
            chunk_emb = chunk_emb.reshape(1, -1)
        sims = chunk_emb @ self._symptom_emb.T

        extracted: Dict[str, float] = {}
        for i in range(len(cleaned)):
            top_idx = np.argsort(-sims[i])[:top_k_per_chunk * 4]
            for idx in top_idx:
                score = float(sims[i, idx])
                if score < config.DICT_SIM_THRESHOLD:
                    break
                cand = self._symptom_list[idx]
                overlap = len(self._symptom_tokens[idx] & query_tokens)
                if overlap < config.DICT_MIN_TOKEN_OVR and score < 0.75:
                    continue
                adj = score + 0.05 * overlap
                if cand not in extracted or extracted[cand] < adj:
                    extracted[cand] = adj

        ranked = sorted(extracted.items(), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:max_total]]

    def embed_symptoms(self, symptoms_list: List[str]) -> List[np.ndarray]:
        """Nhúng triệu chứng dạng batch vector PhoBERT (1 lượt duy nhất)."""
        if not symptoms_list:
            return []
        self._load_embed()
        embs = self._embed_model.encode(
            symptoms_list,
            batch_size=len(symptoms_list),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return [embs[i].astype('float32') for i in range(len(embs))]

    def embed_query_and_symptoms(self, query: str, symptoms_list: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        [TỐI ƯU LATENCY]: Nhúng đồng thời câu query và toàn bộ triệu chứng 
        trong DUY NHẤT 1 lượt batch forward pass của PhoBERT.
        """
        self._load_embed()
        all_texts = [query] + list(symptoms_list)
        all_embs = self._embed_model.encode(
            all_texts,
            batch_size=len(all_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        if all_embs.ndim == 1:
            all_embs = all_embs.reshape(1, -1)
        query_vec = all_embs[0].astype('float32')
        symptom_vecs = [all_embs[i].astype('float32') for i in range(1, len(all_embs))]
        return query_vec, symptom_vecs

    # ============================================================
    #  RAG RETRIEVAL (Mục 4, phương trình 10) - BATCH FAISS & BM25
    # ============================================================
    def retrieve_rag(self, query: str, symptom_vectors: List[np.ndarray],
                     symptoms_list: List[str],
                     precomputed_query_vec: Optional[np.ndarray] = None
                     ) -> Tuple[List[str], Dict[str, float]]:
        """
        Dense (FAISS BATCH per-symptom + query) + Sparse (BM25 expanded) → RRF k=60.
        Tối ưu bằng Batch FAISS Vector Search 1 lượt thay vì vòng lặp tuần tự.
        """
        self._load_embed()
        self._load_faiss_corpus()
        from pyvi import ViTokenizer

        name_col = "disease_name" if "disease_name" in self._df_rag.columns else "ten_benh"

        # 1. Dense: FAISS BATCH search cho toàn bộ vectors trong 1 lệnh duy nhất
        dense_scores: Dict[str, float] = defaultdict(float)
        if self._faiss_index is not None:
            if precomputed_query_vec is not None:
                query_vec = precomputed_query_vec
            else:
                query_vec = self._embed_model.encode(
                    query, normalize_embeddings=True).astype('float32')

            all_vecs = list(symptom_vectors) + [query_vec]
            if all_vecs:
                # Gộp thành ma trận (N, 768) để FAISS search 1 lượt đa luồng
                batch_matrix = np.ascontiguousarray(np.vstack(all_vecs), dtype=np.float32)
                dists_batch, idxs_batch = self._faiss_index.search(
                    batch_matrix, config.DENSE_TOPK_PER_VEC)

                for row_dists, row_idxs in zip(dists_batch, idxs_batch):
                    for dist, idx in zip(row_dists, row_idxs):
                        if idx == -1 or idx >= len(self._df_rag):
                            continue
                        d = str(self._df_rag.iloc[idx][name_col]).strip().lower()
                        if d and d != 'nan':
                            dense_scores[d] += 1.0 / (float(dist) + 1e-5)

        # 2. Sparse: BM25 với query MỞ RỘNG bằng triệu chứng
        exp_query = query + " " + " ".join(symptoms_list)
        tok_query = ViTokenizer.tokenize(exp_query.lower()).split()
        bm25_arr = self._bm25.get_scores(tok_query)
        sparse_scores: Dict[str, float] = defaultdict(float)
        for idx in np.argsort(bm25_arr)[::-1][:config.BM25_TOP_N]:
            if bm25_arr[idx] <= 0:
                continue
            d = str(self._df_rag.iloc[idx][name_col]).strip().lower()
            if d and d != 'nan':
                sparse_scores[d] += bm25_arr[idx]

        # 3. RRF Fusion k=60, phương trình (10)
        k = config.RRF_K
        rrf: Dict[str, float] = defaultdict(float)
        for rank, (d, _) in enumerate(sorted(dense_scores.items(),
                                              key=lambda x: x[1], reverse=True)):
            rrf[d] += 1.0 / (k + rank + 1)
        for rank, (d, _) in enumerate(sorted(sparse_scores.items(),
                                              key=lambda x: x[1], reverse=True)):
            rrf[d] += 1.0 / (k + rank + 1)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:config.TOP_K]
        return [d for d, _ in ranked], dict(ranked)

    # ============================================================
    #  KG RETRIEVAL (Mục 4, phương trình 8-9, 4 tín hiệu Cypher)
    # ============================================================
    def retrieve_kg(self, symptom_vectors: List[np.ndarray],
                    symptoms_list: List[str]
                    ) -> Tuple[List[str], Dict[str, List[str]],
                               Dict[str, float], Dict[str, float]]:
        """
        Truy xuất kênh đồ thị KG qua Neo4j Cypher.
        4 tín hiệu: vector index, bigram CONTAINS, unigram CONTAINS, symptom CONTAINS.
        Port nguyên vẹn từ medkgbert_dx.py retrieve_kg().
        """
        if not config.USE_KG_RETRIEVAL or self._neo4j_driver is None:
            return [], {}, {}, {}

        disease_scores: Dict[str, float] = defaultdict(float)
        disease_max_score: Dict[str, float] = defaultdict(float)
        matched_symptom_count: Dict[str, int] = defaultdict(int)
        kg_subgraphs: Dict[str, List[str]] = defaultdict(list)

        vec_index = config.KG_VECTOR_INDEX
        vec_limit = config.KG_VECTOR_LIMIT
        vec_min = config.KG_VECTOR_SCORE_MIN

        try:
            with self._neo4j_driver.session() as session:
                # 1. Vector index search trên TrieuChung.embedding
                cypher_vec = f"""
                MATCH (t)
                SEARCH t IN (VECTOR INDEX {vec_index} FOR $vec LIMIT {vec_limit}) SCORE AS score
                WHERE score > {vec_min}
                MATCH (t)<-[:CO_TRIEU_CHUNG]-(b:BenhLy)
                RETURN b.ten_benh AS disease, max(score) AS max_score
                """
                for vec in symptom_vectors:
                    try:
                        for rec in session.run(cypher_vec, vec=vec.tolist()):
                            d = rec['disease'].lower()
                            disease_scores[d] += rec['max_score']
                            matched_symptom_count[d] += 1
                            if rec['max_score'] > disease_max_score[d]:
                                disease_max_score[d] = rec['max_score']
                    except Exception:
                        # Fallback nếu chưa có vector index
                        self._fallback_vector_search(
                            session, vec, disease_scores,
                            disease_max_score, matched_symptom_count)
                        break

                # 2. Bigram CONTAINS
                bigrams = set()
                for s in symptoms_list:
                    bigrams.update(_get_bigrams(s))
                if bigrams:
                    cypher_bg = """
                    UNWIND $tokens AS tok
                    MATCH (t:TrieuChung)<-[:CO_TRIEU_CHUNG]-(b:BenhLy)
                    WHERE toLower(t.ten_trieu_chung) CONTAINS tok
                    WITH b, count(DISTINCT t) AS n
                    RETURN b.ten_benh AS disease, n
                    """
                    for rec in session.run(cypher_bg, tokens=list(bigrams)):
                        d = rec['disease'].lower()
                        disease_scores[d] += rec['n'] * 0.25
                        matched_symptom_count[d] += min(rec['n'], 3)

                # 3. Unigram CONTAINS (significant)
                unigrams = set()
                for s in symptoms_list:
                    unigrams.update(_get_significant_unigrams(s))
                if unigrams:
                    for rec in session.run("""
                        UNWIND $tokens AS tok
                        MATCH (t:TrieuChung)<-[:CO_TRIEU_CHUNG]-(b:BenhLy)
                        WHERE toLower(t.ten_trieu_chung) CONTAINS tok
                        WITH b, count(DISTINCT t) AS n
                        RETURN b.ten_benh AS disease, n
                    """, tokens=list(unigrams)):
                        d = rec['disease'].lower()
                        disease_scores[d] += rec['n'] * 0.10
                        matched_symptom_count[d] += min(rec['n'], 2)

                # 4. Subgraph cho top-15
                if disease_scores:
                    top15 = sorted(disease_scores.items(),
                                   key=lambda x: x[1], reverse=True)[:15]
                    for rec in session.run(f"""
                        UNWIND $names AS d_name
                        MATCH (b:BenhLy) WHERE toLower(b.ten_benh) = d_name
                        MATCH (b)-[:CO_TRIEU_CHUNG]->(t:TrieuChung)
                        RETURN b.ten_benh AS disease,
                               collect(t.ten_trieu_chung)[..{config.KG_SYMP_MAX}] AS path
                    """, names=[d for d, _ in top15]):
                        kg_subgraphs[rec['disease'].lower()] = rec['path']

                # 5. Symptom CONTAINS (bypass fragmentation)
                if symptoms_list:
                    for rec in session.run("""
                        UNWIND $symp_list AS symp
                        MATCH (t:TrieuChung)<-[:CO_TRIEU_CHUNG]-(b:BenhLy)
                        WHERE toLower(t.ten_trieu_chung) CONTAINS symp
                           OR symp CONTAINS toLower(t.ten_trieu_chung)
                        WITH b, count(DISTINCT t) AS match_count
                        RETURN b.ten_benh AS disease, match_count
                    """, symp_list=symptoms_list):
                        d = rec['disease'].lower()
                        disease_scores[d] += rec['match_count'] * 0.3
                        matched_symptom_count[d] += rec['match_count']

        except Exception as e:
            logger.error(f"Lỗi KG retrieval: {e}")
            return [], {}, {}, {}

        final = {
            d: matched_symptom_count[d] * disease_max_score.get(d, 0.5)
               + disease_scores[d] * 0.3
            for d in disease_scores
        }
        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:config.TOP_K]
        return ([d for d, _ in ranked], dict(kg_subgraphs),
                dict(ranked), dict(disease_max_score))

    def _fallback_vector_search(self, session, vec, disease_scores,
                                disease_max_score, matched_symptom_count):
        """Fallback khi chưa có Neo4j vector index: dùng matmul."""
        if not hasattr(self, "_warned_fallback"):
            logger.warning(
                f"Chưa có vector index Neo4j — chạy fallback (chậm hơn). "
                f"Hãy tạo: CREATE VECTOR INDEX {config.KG_VECTOR_INDEX} ...")
            self._warned_fallback = True
        if not hasattr(self, "_kg_emb_cache"):
            rows = session.run("""
                MATCH (t:TrieuChung)
                WHERE t.embedding IS NOT NULL AND size(t.embedding) = 768
                MATCH (t)<-[:CO_TRIEU_CHUNG]-(b:BenhLy)
                RETURN t.ten_trieu_chung AS sym, b.ten_benh AS dis, t.embedding AS emb
            """).data()
            names = [r["sym"] for r in rows]
            diseases = [r["dis"] for r in rows]
            emb = np.array([r["emb"] for r in rows], dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self._kg_emb_cache = (names, diseases, emb / norms)
        names, diseases, emb = self._kg_emb_cache
        v = vec / (np.linalg.norm(vec) or 1)
        sims = emb @ v
        top = np.argsort(-sims)[:config.KG_VECTOR_LIMIT]
        for i in top:
            if sims[i] < config.KG_VECTOR_SCORE_MIN:
                continue
            d = diseases[i].lower()
            disease_scores[d] += float(sims[i])
            matched_symptom_count[d] += 1
            if sims[i] > disease_max_score[d]:
                disease_max_score[d] = float(sims[i])

    # ============================================================
    #  BGE RERANKER (Mục 4, phương trình 13-14)
    # ============================================================
    def bge_score_candidates(self, query: str,
                              candidates: List[str]) -> Dict[str, float]:
        """Score ứng viên bằng BGE-reranker-v2-m3 cross-encoder."""
        if not candidates or not config.USE_BGE_RERANKER:
            return {}
        self._load_bge()
        if self._bge_model is None:
            return {}
        import torch
        pairs = [[query, self._disease_to_text.get(d, d)] for d in candidates]
        inputs = self._bge_tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors='pt', max_length=config.BGE_MAX_LENGTH
        ).to(self.device)
        with torch.no_grad():
            scores = self._bge_model(**inputs).logits.view(-1).float().cpu().numpy()
        return dict(zip(candidates, scores.tolist()))

    def bge_rerank(self, query: str, candidates: List[str],
                   top_k: Optional[int] = None
                   ) -> Tuple[List[str], Dict[str, float]]:
        """Tái xếp hạng bằng BGE, trả về top_k ứng viên."""
        top_k = config.RERANK_TOP_N if top_k is None else top_k
        if not candidates:
            return [], {}
        scored = self.bge_score_candidates(query, candidates)
        if not scored:
            return candidates[:top_k], {}
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]], dict(ranked)

    # ============================================================
    #  HYBRID FUSION v11 (Mục 4, phương trình 11/13/15/20)
    # ============================================================
    def hybrid_fusion(self, query: str,
                      rag_ranked: List[str], rag_scores: Dict[str, float],
                      kg_ranked: List[str], kg_scores: Dict[str, float],
                      kg_max_scores: Dict[str, float]
                      ) -> Tuple[List[str], str]:
        """
        Hybrid Fusion v11 — 3 nhánh:
        1. Consensus: top-1 RAG == top-1 KG
        2. KG High Confidence: C_KG >= τ_high
        3. Disagree (KG-anchored): fallback
        Port nguyên vẹn từ medkgbert_dx.py hybrid_fusion_v11().
        """
        rag_n = normalize_scores(rag_scores)
        kg_n = normalize_scores(kg_scores)
        rag_top1 = rag_ranked[0] if rag_ranked else None
        kg_top1 = kg_ranked[0] if kg_ranked else None
        kg_conf = kg_max_scores.get(kg_top1, 0.0) if kg_top1 else 0.0

        # BRANCH 1: CONSENSUS — phương trình (11)
        if rag_top1 and kg_top1 and rag_top1 == kg_top1:
            all_c = set(rag_n) | set(kg_n)
            fused = {}
            for c in all_c:
                r = rag_n.get(c, 0.0)
                k = kg_n.get(c, 0.0)
                s = config.ALPHA_CONSENSUS * r + (1 - config.ALPHA_CONSENSUS) * k
                if r > 0 and k > 0:
                    s += config.CONSENSUS_BONUS
                fused[c] = s
            ranked = sorted(fused.items(), key=lambda x: x[1],
                            reverse=True)[:config.TOP_K]
            return [d for d, _ in ranked], "consensus"

        # BRANCH 2: KG HIGH CONFIDENCE — phương trình (13)
        if (kg_conf >= config.KG_HIGH_CONF_THRESHOLD
                and len(kg_ranked) >= config.KG_HIGH_CONF_MIN_CANDIDATES):
            pool = kg_ranked[:10]
            bge_scores = self.bge_score_candidates(query, pool)
            rag_top5 = set(rag_ranked[:5])
            rerank = {}
            for d in pool:
                bge_norm = _sigmoid(bge_scores.get(d, 0.0)) if bge_scores else 0.0
                kg_s = kg_n.get(d, 0.0)
                s = 0.55 * bge_norm + 0.45 * kg_s
                if d in rag_top5:
                    s += 0.20
                rerank[d] = s
            ranked = sorted(rerank.items(), key=lambda x: x[1],
                            reverse=True)[:config.TOP_K]
            return [d for d, _ in ranked], "kg_high_conf"

        # BRANCH 3: DISAGREE (KG-anchored) — phương trình (20)
        pool, seen = [], set()
        for d in kg_ranked[:10]:
            if d not in seen:
                pool.append(d)
                seen.add(d)
        for d in rag_ranked[:3]:
            if d not in seen:
                pool.append(d)
                seen.add(d)
        bge_scores = self.bge_score_candidates(query, pool)
        rag_top3 = set(rag_ranked[:3])
        kg_top3 = set(kg_ranked[:3])
        rerank = {}
        for d in pool:
            bge_norm = _sigmoid(bge_scores.get(d, 0.0)) if bge_scores else 0.0
            kg_s = kg_n.get(d, 0.0)
            rag_s = rag_n.get(d, 0.0)
            s = 0.40 * bge_norm + 0.40 * kg_s + 0.20 * rag_s
            if d in kg_top3 and d in rag_top3:
                s += 0.30
            elif d in kg_top3:
                s += 0.10
            elif d in rag_top3 and d not in kg_ranked[:5]:
                s -= 0.05
            if kg_conf > config.KG_MED_CONF_THRESHOLD and d == kg_top1:
                s += 0.15
            rerank[d] = s
        ranked = sorted(rerank.items(), key=lambda x: x[1],
                        reverse=True)[:config.TOP_K]
        return [d for d, _ in ranked], "disagree_kg_anchored"

    # ============================================================
    #  RUN PIPELINE (endpoint cho web)
    # ============================================================
    def run_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline MedKG-HRR chuẩn bài báo v16.
        
        Returns:
            dict với: symptoms_extracted, rag_results, kg_results,
                      fusion_mode, ranked_diseases, latency_ms, pipeline_info
        """
        if not self.is_ready:
            return {"error": "MedKG-HRR chưa khởi tạo", "ranked_diseases": []}

        t0 = time.perf_counter()

        # 1. Trích xuất triệu chứng (Mục 4, bước (i))
        symptoms = self.extract_symptoms(query)
        if not symptoms:
            symptoms = [query]  # Fallback: dùng query thô
        t_extract = (time.perf_counter() - t0) * 1000

        # 2. [BATCH EMBEDDING]: Nhúng đồng thời query + toàn bộ triệu chứng trong 1 batch duy nhất
        t1 = time.perf_counter()
        query_vec, vectors = self.embed_query_and_symptoms(query, symptoms)
        t_embed = (time.perf_counter() - t1) * 1000

        # 3. RAG retrieval (FAISS Batch Dense + BM25 Sparse → RRF)
        t2 = time.perf_counter()
        rag_ranked, rag_scores = self.retrieve_rag(
            query, vectors, symptoms, precomputed_query_vec=query_vec)
        t_rag = (time.perf_counter() - t2) * 1000

        # 4. KG retrieval
        t3 = time.perf_counter()
        kg_ranked, kg_subgraphs, kg_scores, kg_max = self.retrieve_kg(
            vectors, symptoms)
        t_kg = (time.perf_counter() - t3) * 1000

        # 5. Hybrid Fusion v11
        t4 = time.perf_counter()
        if kg_ranked:
            hybrid_ranked, fusion_mode = self.hybrid_fusion(
                query, rag_ranked, rag_scores, kg_ranked, kg_scores, kg_max)
        else:
            # Không có KG → Retrieval-only + BGE rerank
            hybrid_ranked, _ = self.bge_rerank(query, rag_ranked, config.TOP_K)
            if not hybrid_ranked:
                hybrid_ranked = rag_ranked
            fusion_mode = "retrieval_only"
        t_hybrid = (time.perf_counter() - t4) * 1000

        # 6. BGE rerank final cho kết quả hybrid
        t5 = time.perf_counter()
        if fusion_mode == "consensus":
            # Consensus bỏ qua reranking (theo bài báo Mục 4)
            final_ranked = hybrid_ranked
        else:
            final_ranked_bge, _ = self.bge_rerank(query, hybrid_ranked, config.TOP_K)
            final_ranked = final_ranked_bge if final_ranked_bge else hybrid_ranked
        t_rerank = (time.perf_counter() - t5) * 1000

        total_ms = (time.perf_counter() - t0) * 1000

        # 7. Gắn siêu dữ liệu (ICD-10, chuyên khoa, triệu chứng)
        results = []
        for rank, d_lower in enumerate(final_ranked, 1):
            info = self._disease_dict.get(d_lower, {})
            results.append({
                "rank": rank,
                "disease": info.get("name", d_lower.title()),
                "icd10": info.get("icd10", ""),
                "icd10_name": info.get("icd10_name", ""),
                "specialty": info.get("specialty", "Đa khoa / Nội khoa"),
                "symptom": info.get("symptoms", ""),
                "score": rag_scores.get(d_lower, 0.0),
                "kg_score": kg_scores.get(d_lower, 0.0),
                "kg_evidence": kg_subgraphs.get(d_lower, []),
                "source": self._determine_source(
                    d_lower, rag_ranked, kg_ranked, fusion_mode),
            })

        return {
            "symptoms_extracted": symptoms,
            "fusion_mode": fusion_mode,
            "ranked_diseases": results,
            "latency_ms": {
                "extract": round(t_extract, 1),
                "embed": round(t_embed, 1),
                "rag": round(t_rag, 1),
                "kg": round(t_kg, 1),
                "hybrid_fusion": round(t_hybrid, 1),
                "rerank": round(t_rerank, 1),
                "total": round(total_ms, 1),
            },
            "pipeline_info": {
                "components": dict(self._components_status),
                "config": {
                    "top_k": config.TOP_K,
                    "rrf_k": config.RRF_K,
                    "alpha_consensus": config.ALPHA_CONSENSUS,
                    "kg_high_conf_threshold": config.KG_HIGH_CONF_THRESHOLD,
                }
            }
        }

    def _determine_source(self, disease: str, rag_ranked: List[str],
                          kg_ranked: List[str], fusion_mode: str) -> str:
        """Xác định nguồn ứng viên (cho UI hiển thị)."""
        in_rag = disease in rag_ranked[:5]
        in_kg = disease in kg_ranked[:5]
        if in_rag and in_kg:
            return "KG + RAG"
        elif in_kg:
            return "KG"
        elif in_rag:
            return "RAG"
        return fusion_mode

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Trả về trạng thái các components cho endpoint /api/pipeline-info."""
        return {
            "is_ready": self.is_ready,
            "components": dict(self._components_status),
            "architecture": "MedKG-HRR v16 (3 kênh + 3 nhánh)",
            "paper_reference": "Mục 4, Phụ lục D",
            "channels": {
                "kg": "Neo4j Cypher (4 tín hiệu)" if config.USE_KG_RETRIEVAL else "disabled",
                "dense": "FAISS PhoBERT contrastive (per-symptom)",
                "sparse": "BM25 Okapi (expanded query)",
            },
            "fusion": "Hybrid Fusion v11: Consensus / KG High Conf / Disagree",
            "reranker": "BGE-reranker-v2-m3" if config.USE_BGE_RERANKER else "disabled",
        }

    def close(self):
        if self._neo4j_driver:
            self._neo4j_driver.close()
            self._neo4j_driver = None


# Global singleton engine
medkghrr_engine = MedKGHRREngine()

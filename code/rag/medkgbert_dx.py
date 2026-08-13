"""
MedKGBERT-Dx Pipeline v11 (modular port từ notebook eval(2).ipynb)
==================================================================
Module hoá để gọi từ eval_rag.py và các script khác.

Thành phần:
  - SegmentedEncoder (pyvi wrap PhoBERT)
  - Symptom Extraction v10 (Dict + Llama CoT fallback)
  - RAG retrieval: FAISS dense + BM25 sparse → RRF
  - KG retrieval: Vector index + bigram CONTAINS + unigram CONTAINS + symptom CONTAINS
  - BGE-Reranker-v2-m3 (cross-encoder)
  - LLM cross-modal reranker (Llama-3-8B Instruct)
  - Hybrid Fusion v11 (3 nhánh: consensus / kg_high_conf / disagree_kg_anchored)

Mỗi component lazy-load — nếu thiếu GPU/model, tự skip có cảnh báo.

Usage:
    from code.rag.medkgbert_dx import MedKGBertDx, Config

    pipe = MedKGBertDx(Config())
    out  = pipe.run_all("Tôi đau đầu kèm sốt 3 ngày...")
    # out = {
    #   'symptoms':       [...],
    #   'rag':            (ranked, scores),
    #   'kg':             (ranked, subgraphs, scores, max_scores),
    #   'hybrid':         (ranked, fusion_mode),
    #   'rag_top1':       'tên bệnh',
    #   'kg_top1':        '...',
    #   'hybrid_top1':    '...',
    #   'latency_ms':     {...},
    # }
"""
from __future__ import annotations

import os
import re
import time
import warnings
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


# ============================================================
#  CONFIG
# ============================================================
@dataclass
class Config:
    # Neo4j (local V2)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )

    # Model paths
    # PhoBERT-base tinh chỉnh tương phản (Mục 5.4). Bản C:\ gốc mất khi cài lại Win;
    # bản trong sao lưu và bản trong D:\Download\Medical_NLP\ trùng SHA-256.
    PHOBERT_PATH: str = r"C:\BACKUP_RESEARCH_2026-08-11\models\phobert-medical-constractive"
    BGE_RERANKER_PATH: str = "BAAI/bge-reranker-v2-m3"
    LLAMA_MODEL_PATH: str = "NousResearch/Meta-Llama-3-8B-Instruct"

    # Data paths — toàn bộ đã khôi phục vào data/, xem data/ASSETS_MANIFEST.md
    CSV_TEST_FILE: str = "data/test_300_paraphrased.csv"
    CSV_MEDICAL: str = "data/medical_data_cleaned.csv"
    # Corpus RAG: 8.214 đoạn. Trên Drive của hienhocit205@ tệp này mang tên
    # rag_chunks.csv — cùng SHA-256, chỉ khác tên.
    CSV_RAG_CHUNKS: str = "data/medical_chunks_hierarchical.csv"
    # IndexFlatIP, 768 chiều, ntotal = 8.214 — khớp 1:1 theo dòng với CSV_RAG_CHUNKS
    # (retrieve_rag tra ngược bằng df.iloc[idx], nên hai tệp phải luôn đi cùng cặp).
    FAISS_INDEX_PATH: str = "data/faiss_medical_local_phobert.index"

    # Retrieval — Phụ lục D, mục I "Tham số thiết lập đánh giá"
    TOP_K: int = 10              # Retrieval Depth (TOP_K) = 10
    RRF_K: int = 60              # RRF Constant (k) = 60, phương trình (10)
    RERANK_TOP_N: int = 5        # Số ứng viên tối đa đưa vào BGE-reranker
    BGE_MAX_LENGTH: int = 1024   # Rerank Max Window Length (VN) = 1024
    DENSE_TOPK_PER_VEC: int = 5  # Số láng giềng FAISS lấy cho mỗi vector triệu chứng
    BM25_TOP_N: int = 50         # Độ sâu danh sách BM25 trước khi gộp RRF
    KG_SYMP_MAX: int = 6         # KG_SYMP_MAX = 6 triệu chứng/bệnh khi dựng subgraph
    DEVICE: str = "cuda"  # auto-fallback to cpu

    # Symptom extraction
    DICT_SIM_THRESHOLD: float = 0.55
    DICT_MIN_TOKEN_OVR: int = 1
    DICT_MIN_SYM_LEN: int = 3
    DICT_MAX_SYM_LEN: int = 80
    EXTRACT_MIN_DICT_COUNT: int = 2

    # Hybrid v11 — Phụ lục D mục II, phương trình (11) (13) (15) (20)
    ALPHA_CONSENSUS: float = 0.25       # eq (11): 0,25 sRAG + 0,75 sKG
    CONSENSUS_BONUS: float = 0.40       # eq (11): Consensus Bonus Score = 0,40
    KG_HIGH_CONF_THRESHOLD: float = 0.85  # τ_high, eq (15)
    KG_MED_CONF_THRESHOLD: float = 0.65   # τ_med, eq (20)

    # LƯU Ý ĐỐI SOÁT: điều kiện này KHÔNG có trong phương trình (15) của bài báo.
    # Bài viết mode = KG High Confidence khi d_KG ≠ d_RAG và C_KG ≥ τ_high, không kèm
    # ràng buộc về số ứng viên. Lượt chạy of-record (87,67%) lại có ràng buộc này.
    # Giữ nguyên giá trị 3 để tái lập đúng số đã công bố; cần bổ sung vào bài báo.
    KG_HIGH_CONF_MIN_CANDIDATES: int = 3

    # KG vector index
    KG_VECTOR_INDEX: str = "trieu_chung_vector_index"
    KG_VECTOR_LIMIT: int = 8
    KG_VECTOR_SCORE_MIN: float = 0.45

    # Toggles — sẽ tự tắt nếu thiếu CUDA / model
    USE_BGE_RERANKER: bool = True
    USE_LLAMA_COT_FALLBACK: bool = True

    # Mục 4 và ghi chú Bảng 7: cấu hình được báo cáo KHÔNG dùng bộ tái xếp hạng
    # đa phương thức bằng LLM ("bỏ bộ chọn Llama-3-8B ở các đường cơ sở").
    # Mặc định cũ là True → chạy ra số KHÁC với lượt đối xứng đã công bố.
    # Đặt False để mặc định tái lập đúng Bảng 7.
    USE_LLM_CROSS_RERANK: bool = False


# ============================================================
#  STOPWORDS / REGEX (giữ nguyên notebook)
# ============================================================
_STOP_DICT = {'và','hoặc','của','là','có','bị','rất','nhiều','ít','này','đó',
              'như','nên','phải','tôi','bạn','các','một','hai','khi','lúc',
              'cảm','thấy','hay','thường','dạo','gần','đây','mấy','ngày','hôm'}
_KG_STOP = {'cảm','thấy','bị','và','hoặc','rất','nhiều','ít',
            'tôi','của','là','có','được','mất','đã','sẽ',
            'này','đó','như','nên','phải','cũng','còn',
            'hay','thường','đôi','khi','lúc','một','hai'}
_KG_GENERIC = {'đau','sốt','ho','mệt','nôn','ngứa','sưng','rát'}

_PREAMBLE_RE = re.compile(
    r'(?i)'
    r'(here\s+is\s+(the\s+)?(list\s+of\s+)?symptoms?[^\n,]*[:\n]+)'
    r'|(danh\s+sách\s+(các\s+)?triệu\s+chứng[^\n,]*[:\n]+)'
    r'|(triệu\s+chứng[^\n,]*?:\s*)'
    r'|(symptoms?\s*:\s*)'
)
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
#  SEGMENTED ENCODER (v10)
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
#  PIPELINE
# ============================================================
class MedKGBertDx:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self._device = None
        self._embed_model = None        # SegmentedEncoder
        self._raw_embed = None
        self._bge_tokenizer = None
        self._bge_model = None
        self._llm_tokenizer = None
        self._llm_model = None
        self._faiss_index = None
        self._driver = None
        self._df_rag = None
        self._corpus = None
        self._bm25 = None
        self._disease_to_text: Dict[str, str] = {}
        self._symptom_list: List[str] = []
        self._symptom_emb: Optional[np.ndarray] = None
        self._symptom_tokens: List[set] = []

    # ---------- device ----------
    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
            if self._device == "cpu":
                # GPU-heavy components tự tắt
                self.cfg.USE_BGE_RERANKER = False
                self.cfg.USE_LLAMA_COT_FALLBACK = False
                self.cfg.USE_LLM_CROSS_RERANK = False
                print("⚠ CUDA không khả dụng → tắt BGE rerank + Llama rerank.")
        return self._device

    # ---------- model loaders (lazy) ----------
    def _load_embed(self):
        if self._embed_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        from pyvi import ViTokenizer
        print(f"[load] PhoBERT contrastive: {self.cfg.PHOBERT_PATH}")
        self._raw_embed = SentenceTransformer(self.cfg.PHOBERT_PATH, device=self.device)
        self._embed_model = SegmentedEncoder(self._raw_embed, ViTokenizer)

    def _load_bge(self):
        if self._bge_model is not None or not self.cfg.USE_BGE_RERANKER:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"[load] BGE reranker: {self.cfg.BGE_RERANKER_PATH}")
            self._bge_tokenizer = AutoTokenizer.from_pretrained(self.cfg.BGE_RERANKER_PATH)
            self._bge_model = AutoModelForSequenceClassification.from_pretrained(
                self.cfg.BGE_RERANKER_PATH, torch_dtype=torch.float16
            ).to(self.device)
            self._bge_model.eval()
        except Exception as e:
            print(f"⚠ Không load được BGE ({e}) — tắt BGE rerank.")
            self.cfg.USE_BGE_RERANKER = False

    def _load_llm(self):
        if self._llm_model is not None or (
            not self.cfg.USE_LLAMA_COT_FALLBACK and not self.cfg.USE_LLM_CROSS_RERANK
        ):
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            print(f"[load] Llama-3-8B (4-bit): {self.cfg.LLAMA_MODEL_PATH}")
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self._llm_tokenizer = AutoTokenizer.from_pretrained(self.cfg.LLAMA_MODEL_PATH)
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                self.cfg.LLAMA_MODEL_PATH, quantization_config=bnb, device_map="auto"
            )
        except Exception as e:
            print(f"⚠ Không load được Llama ({e}) — tắt Llama rerank/CoT.")
            self.cfg.USE_LLAMA_COT_FALLBACK = False
            self.cfg.USE_LLM_CROSS_RERANK = False

    def _load_neo4j(self):
        if self._driver is not None:
            return
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(
            self.cfg.NEO4J_URI, auth=(self.cfg.NEO4J_USER, self.cfg.NEO4J_PASSWORD)
        )

    def _load_faiss_corpus(self):
        if self._bm25 is not None:
            return
        from pyvi import ViTokenizer
        import faiss
        # FAISS index
        if os.path.exists(self.cfg.FAISS_INDEX_PATH):
            print(f"[load] FAISS index: {self.cfg.FAISS_INDEX_PATH}")
            self._faiss_index = faiss.read_index(self.cfg.FAISS_INDEX_PATH)
        else:
            print(f"⚠ Không tìm thấy {self.cfg.FAISS_INDEX_PATH} — RAG sẽ chỉ dùng BM25.")
            self._faiss_index = None

        # RAG corpus
        if not os.path.exists(self.cfg.CSV_RAG_CHUNKS):
            # Fallback: dùng medical_data_cleaned.csv để dựng disease_to_text
            print(f"⚠ Không thấy {self.cfg.CSV_RAG_CHUNKS} — fallback {self.cfg.CSV_MEDICAL}")
            path = self.cfg.CSV_MEDICAL
        else:
            path = self.cfg.CSV_RAG_CHUNKS
        try:
            self._df_rag = pd.read_csv(path, on_bad_lines='skip', engine='python', encoding='utf-8')
            if self._df_rag.shape[1] == 1:
                self._df_rag = pd.read_csv(path, delimiter=';', on_bad_lines='skip',
                                           engine='python', encoding='utf-8')
        except Exception:
            self._df_rag = pd.read_csv(path, delimiter=';', on_bad_lines='skip',
                                       engine='python', encoding='utf-8')

        self._corpus = self._df_rag.fillna('').astype(str).agg(' '.join, axis=1).tolist()
        from rank_bm25 import BM25Okapi
        tokenized = [ViTokenizer.tokenize(doc.lower()).split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

        # disease_to_text
        name_col = "disease_name" if "disease_name" in self._df_rag.columns \
            else ("ten_benh" if "ten_benh" in self._df_rag.columns else None)
        if name_col:
            for idx, row in self._df_rag.iterrows():
                d = str(row[name_col]).strip().lower()
                if d and d != 'nan':
                    self._disease_to_text[d] = self._disease_to_text.get(d, "") + " " + self._corpus[idx]

    # ============================================================
    #  SYMPTOM DICTIONARY
    # ============================================================
    def build_symptom_dictionary(self, batch_size: int = 128):
        """Build từ CSV_RAG_CHUNKS (positive_symptoms_norm) — gọi 1 lần trước retrieve."""
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
                    if self.cfg.DICT_MIN_SYM_LEN <= len(s) <= self.cfg.DICT_MAX_SYM_LEN:
                        all_symps.add(s)
        self._symptom_list = sorted(all_symps)
        if not self._symptom_list:
            raise RuntimeError("Không build được symptom dictionary — CSV thiếu cột positive_symptoms[_norm].")
        print(f"[dict] Symptom dictionary: {len(self._symptom_list)} unique")
        self._symptom_emb = self._embed_model.encode(
            self._symptom_list, batch_size=batch_size,
            show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )
        self._symptom_tokens = [_toks(s) for s in self._symptom_list]

    # ============================================================
    #  v9 — SYMPTOM EXTRACTION
    # ============================================================
    def extract_symptoms_dict(self, query: str, top_k_per_chunk: int = 3,
                              max_total: int = 8) -> List[str]:
        self.build_symptom_dictionary()
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
                if score < self.cfg.DICT_SIM_THRESHOLD:
                    break
                cand = self._symptom_list[idx]
                overlap = len(self._symptom_tokens[idx] & query_tokens)
                if overlap < self.cfg.DICT_MIN_TOKEN_OVR and score < 0.75:
                    continue
                adj = score + 0.05 * overlap
                if cand not in extracted or extracted[cand] < adj:
                    extracted[cand] = adj

        ranked = sorted(extracted.items(), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:max_total]]

    _COT_PROMPT = """Bạn là bác sĩ trợ lý chuyên trích xuất TRIỆU CHỨNG y khoa từ lời kể của bệnh nhân.

QUY TẮC:
1. Chỉ trả về triệu chứng — KHÔNG tự thêm bệnh, KHÔNG suy đoán nguyên nhân.
2. Bỏ qua từ cảm thán ("bác sĩ ơi", "tôi muốn hỏi", "dạo này"...).
3. Chuẩn hóa thành cụm danh từ ngắn (vd: "đau kinh khủng ở bụng" → "đau bụng dữ dội").
4. Nếu không tìm thấy triệu chứng rõ ràng → trả về [].

VÍ DỤ (One-shot):
Lời kể: "Bác sĩ ơi, mấy hôm nay tôi hay bị đau đầu kinh khủng và buồn nôn. Tôi cũng thấy chóng mặt và mệt lả người."

Suy luận (Chain-of-Thought):
- "Bác sĩ ơi, mấy hôm nay tôi hay bị" → cảm thán, bỏ qua.
- "đau đầu kinh khủng" → triệu chứng: đau đầu dữ dội.
- "buồn nôn" → triệu chứng: buồn nôn.
- "Tôi cũng thấy" → cảm thán, bỏ qua.
- "chóng mặt" → triệu chứng: chóng mặt.
- "mệt lả người" → triệu chứng: mệt mỏi.

Kết quả JSON: {"symptoms": ["đau đầu dữ dội", "buồn nôn", "chóng mặt", "mệt mỏi"]}

---
Lời kể: "{query}"

Suy luận (Chain-of-Thought):"""

    def extract_symptoms_llama_cot(self, query: str, max_new_tokens: int = 256) -> List[str]:
        if not self.cfg.USE_LLAMA_COT_FALLBACK:
            return []
        self._load_llm()
        if self._llm_model is None:
            return []
        import torch
        prompt = self._COT_PROMPT.replace("{query}", query)
        messages = [
            {"role": "system", "content":
                "You are a medical symptom extractor. Follow the Chain-of-Thought "
                "then output a JSON with key 'symptoms'."},
            {"role": "user", "content": prompt}
        ]
        prompt_str = self._llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._llm_tokenizer(prompt_str, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._llm_model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self._llm_tokenizer.eos_token_id
            )
        text = self._llm_tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True)
        m = re.search(r'\{\s*"symptoms"\s*:\s*\[(.*?)\]\s*\}', text, re.DOTALL)
        if m:
            items = re.findall(r'"([^"]+)"', m.group(1))
            raw = [s.strip().lower() for s in items if 2 < len(s.strip()) < 80]
        else:
            finds = re.findall(r'triệu chứng:\s*([^\n.{}]+)', text, re.IGNORECASE)
            raw = [f.strip().lower().rstrip('.,') for f in finds
                   if 2 < len(f.strip()) < 80]
        if not raw:
            return []
        llama_emb = self._embed_model.encode(raw, convert_to_numpy=True,
                                             normalize_embeddings=True)
        if llama_emb.ndim == 1:
            llama_emb = llama_emb.reshape(1, -1)
        sims = llama_emb @ self._symptom_emb.T
        mapped, seen = [], set()
        for i in range(len(raw)):
            idx = int(np.argmax(sims[i])); score = float(sims[i, idx])
            if score >= 0.60:
                cand = self._symptom_list[idx]
                if cand not in seen:
                    seen.add(cand); mapped.append(cand)
        return mapped[:8]

    def extract_symptoms(self, query: str) -> List[str]:
        """v10 — dict trước, Llama CoT fallback nếu dict < min_count."""
        dict_symps = self.extract_symptoms_dict(query)
        if len(dict_symps) >= self.cfg.EXTRACT_MIN_DICT_COUNT:
            return dict_symps
        extra = self.extract_symptoms_llama_cot(query)
        seen = set(dict_symps)
        return dict_symps + [s for s in extra if s not in seen]

    def embed_symptoms(self, symptoms_list: List[str]) -> List[np.ndarray]:
        self._load_embed()
        return [self._embed_model.encode(s, normalize_embeddings=True).astype('float32')
                for s in symptoms_list]

    # ============================================================
    #  RAG (FAISS + BM25 → RRF)
    # ============================================================
    def retrieve_rag(self, query: str, symptom_vectors: List[np.ndarray],
                     symptoms_list: List[str]) -> Tuple[List[str], Dict[str, float]]:
        self._load_embed()
        self._load_faiss_corpus()
        from pyvi import ViTokenizer

        name_col = "disease_name" if "disease_name" in self._df_rag.columns else "ten_benh"

        dense_scores: Dict[str, float] = defaultdict(float)
        if self._faiss_index is not None:
            query_vec = self._embed_model.encode(query, normalize_embeddings=True).astype('float32')
            all_vecs = list(symptom_vectors) + [query_vec]
            for vec in all_vecs:
                dists, idxs = self._faiss_index.search(np.array([vec]),
                                                       self.cfg.DENSE_TOPK_PER_VEC)
                for i, idx in enumerate(idxs[0]):
                    if idx == -1:
                        continue
                    d = str(self._df_rag.iloc[idx][name_col]).strip().lower()
                    if d and d != 'nan':
                        dense_scores[d] += 1.0 / (dists[0][i] + 1e-5)

        exp_query = query + " " + " ".join(symptoms_list)
        tok_query = ViTokenizer.tokenize(exp_query.lower()).split()
        bm25_arr = self._bm25.get_scores(tok_query)
        sparse_scores: Dict[str, float] = defaultdict(float)
        for idx in np.argsort(bm25_arr)[::-1][:self.cfg.BM25_TOP_N]:
            if bm25_arr[idx] <= 0:
                continue
            d = str(self._df_rag.iloc[idx][name_col]).strip().lower()
            if d and d != 'nan':
                sparse_scores[d] += bm25_arr[idx]

        # eq (10): sRAG(d|q) = Σ_c 1 / (k + rank_c(d|q)), k = 60
        k = self.cfg.RRF_K
        rrf: Dict[str, float] = defaultdict(float)
        for rank, (d, _) in enumerate(sorted(dense_scores.items(),
                                             key=lambda x: x[1], reverse=True)):
            rrf[d] += 1.0 / (k + rank + 1)
        for rank, (d, _) in enumerate(sorted(sparse_scores.items(),
                                             key=lambda x: x[1], reverse=True)):
            rrf[d] += 1.0 / (k + rank + 1)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:self.cfg.TOP_K]
        return [d for d, _ in ranked], dict(ranked)

    # ============================================================
    #  KG retrieval (v8 — 4 signals)
    # ============================================================
    def retrieve_kg(self, symptom_vectors: List[np.ndarray],
                    symptoms_list: List[str]
                    ) -> Tuple[List[str], Dict[str, List[str]],
                               Dict[str, float], Dict[str, float]]:
        self._load_neo4j()
        disease_scores: Dict[str, float] = defaultdict(float)
        disease_max_score: Dict[str, float] = defaultdict(float)
        matched_symptom_count: Dict[str, int] = defaultdict(int)
        kg_subgraphs: Dict[str, List[str]] = defaultdict(list)

        vec_index = self.cfg.KG_VECTOR_INDEX
        vec_limit = self.cfg.KG_VECTOR_LIMIT
        vec_min = self.cfg.KG_VECTOR_SCORE_MIN

        with self._driver.session() as session:
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
                except Exception as e:
                    # Fallback nếu chưa có vector index — dùng cosine matmul
                    self._fallback_vector_search(session, vec, disease_scores,
                                                 disease_max_score, matched_symptom_count)
                    # đã in cảnh báo trong fallback
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

            # 4. Subgraph cho top-15 để present
            if disease_scores:
                top15 = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:15]
                for rec in session.run(f"""
                    UNWIND $names AS d_name
                    MATCH (b:BenhLy) WHERE toLower(b.ten_benh) = d_name
                    MATCH (b)-[:CO_TRIEU_CHUNG]->(t:TrieuChung)
                    RETURN b.ten_benh AS disease,
                           collect(t.ten_trieu_chung)[..{self.cfg.KG_SYMP_MAX}] AS path
                """, names=[d for d, _ in top15]):
                    kg_subgraphs[rec['disease'].lower()] = rec['path']

            # 5. Symptom CONTAINS (bypass fragmentation final)
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

        final = {
            d: matched_symptom_count[d] * disease_max_score.get(d, 0.5)
               + disease_scores[d] * 0.3
            for d in disease_scores
        }
        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:self.cfg.TOP_K]
        return [d for d, _ in ranked], dict(kg_subgraphs), dict(ranked), dict(disease_max_score)

    def _fallback_vector_search(self, session, vec, disease_scores,
                                disease_max_score, matched_symptom_count):
        """Khi chưa CREATE VECTOR INDEX, dùng matmul Cypher (chậm hơn nhưng chạy được)."""
        if not hasattr(self, "_warned_fallback"):
            print("⚠ Chưa có vector index Neo4j — chạy fallback (chậm hơn). "
                  f"Hãy tạo: CREATE VECTOR INDEX {self.cfg.KG_VECTOR_INDEX} "
                  "FOR (t:TrieuChung) ON t.embedding "
                  "OPTIONS {indexConfig: {`vector.dimensions`: 768, "
                  "`vector.similarity_function`: 'cosine'}}")
            self._warned_fallback = True
        # Đọc toàn bộ embedding 1 lần
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
            norms = np.linalg.norm(emb, axis=1, keepdims=True); norms[norms == 0] = 1
            self._kg_emb_cache = (names, diseases, emb / norms)
        names, diseases, emb = self._kg_emb_cache
        v = vec / (np.linalg.norm(vec) or 1)
        sims = emb @ v
        top = np.argsort(-sims)[: self.cfg.KG_VECTOR_LIMIT]
        for i in top:
            if sims[i] < self.cfg.KG_VECTOR_SCORE_MIN:
                continue
            d = diseases[i].lower()
            disease_scores[d] += float(sims[i])
            matched_symptom_count[d] += 1
            if sims[i] > disease_max_score[d]:
                disease_max_score[d] = float(sims[i])

    # ============================================================
    #  BGE Reranker
    # ============================================================
    def bge_score_candidates(self, query: str, candidates: List[str]) -> Dict[str, float]:
        if not candidates or not self.cfg.USE_BGE_RERANKER:
            return {}
        self._load_bge()
        if self._bge_model is None:
            return {}
        import torch
        pairs = [[query, self._disease_to_text.get(d, d)] for d in candidates]
        inputs = self._bge_tokenizer(pairs, padding=True, truncation=True,
                                     return_tensors='pt',
                                     max_length=self.cfg.BGE_MAX_LENGTH).to(self.device)
        with torch.no_grad():
            scores = self._bge_model(**inputs).logits.view(-1).float().cpu().numpy()
        return dict(zip(candidates, scores.tolist()))

    def bge_rerank(self, query: str, candidates: List[str], top_k: Optional[int] = None
                   ) -> Tuple[List[str], Dict[str, float]]:
        top_k = self.cfg.RERANK_TOP_N if top_k is None else top_k
        if not candidates:
            return [], {}
        scored = self.bge_score_candidates(query, candidates)
        if not scored:
            return candidates[:top_k], {}
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]], dict(ranked)

    # ============================================================
    #  LLM Cross-modal Reranker
    # ============================================================
    def llm_cross_rerank(self, symptoms_list: List[str], candidates: List[str],
                         kg_subgraphs: Dict[str, List[str]]) -> List[str]:
        if not candidates:
            return []
        if len(candidates) == 1 or not self.cfg.USE_LLM_CROSS_RERANK:
            return [candidates[0]]
        self._load_llm()
        if self._llm_model is None:
            return [candidates[0]]
        import torch
        from rapidfuzz import fuzz, process

        bge_top1 = candidates[0]
        bge_top3set = set(candidates[:3])
        symp_str = ", ".join(symptoms_list)
        modal = ""
        for c in candidates:
            text_ev = self._disease_to_text.get(c, "Không có thông tin")[:180]
            path = kg_subgraphs.get(c, [])
            graph_ev = ("KG: " + ", ".join(path)) if path else "Không có data KG"
            modal += f"[{c}]\n  • Y văn: {text_ev}...\n  • {graph_ev}\n\n"

        prompt = (
            "Bạn là bác sĩ AI hội chẩn.\n\n"
            f"Triệu chứng bệnh nhân: {symp_str}\n\n"
            "Bệnh ứng viên (chọn 1):\n"
            + "\n".join(f"- {c}" for c in candidates)
            + "\n\nBằng chứng:\n" + modal
            + "Quy tắc:\n"
            "1. Đếm triệu chứng bệnh nhân trùng với KG của mỗi bệnh.\n"
            "2. Chọn bệnh có nhiều triệu chứng trùng + mô tả phù hợp nhất.\n"
            "3. Output BẮT BUỘC: <KetLuan>tên bệnh</KetLuan>\n\nKHÔNG giải thích."
        )
        messages = [
            {"role": "system", "content":
                "You are a precise medical AI. Output one disease inside <KetLuan> tags."},
            {"role": "user", "content": prompt}
        ]
        prompt_str = self._llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._llm_tokenizer(prompt_str, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._llm_model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
                pad_token_id=self._llm_tokenizer.eos_token_id
            )
        ans = self._llm_tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                         skip_special_tokens=True)
        m = re.search(r'<KetLuan>(.*?)</KetLuan>', ans, re.IGNORECASE)
        if m:
            final_d = m.group(1).strip().lower()
            hit = process.extractOne(final_d, candidates, scorer=fuzz.WRatio)
            if hit and hit[1] >= 65:
                pick = hit[0]
                if pick in bge_top3set:
                    return [pick]
        return [bge_top1]

    # ============================================================
    #  HYBRID FUSION v11 (3 nhánh)
    # ============================================================
    def hybrid_fusion_v11(self, query: str,
                          rag_ranked: List[str], rag_scores: Dict[str, float],
                          kg_ranked: List[str], kg_scores: Dict[str, float],
                          kg_max_scores: Dict[str, float]
                          ) -> Tuple[List[str], str]:
        rag_n = normalize_scores(rag_scores)
        kg_n = normalize_scores(kg_scores)
        rag_top1 = rag_ranked[0] if rag_ranked else None
        kg_top1 = kg_ranked[0] if kg_ranked else None
        kg_conf = kg_max_scores.get(kg_top1, 0.0) if kg_top1 else 0.0

        # BRANCH 1: CONSENSUS
        if rag_top1 and kg_top1 and rag_top1 == kg_top1:
            all_c = set(rag_n) | set(kg_n)
            fused = {}
            for c in all_c:
                r = rag_n.get(c, 0.0); k = kg_n.get(c, 0.0)
                s = self.cfg.ALPHA_CONSENSUS * r + (1 - self.cfg.ALPHA_CONSENSUS) * k
                if r > 0 and k > 0:
                    s += self.cfg.CONSENSUS_BONUS
                fused[c] = s
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:self.cfg.TOP_K]
            return [d for d, _ in ranked], "consensus"

        # BRANCH 2: KG HIGH CONF
        # eq (15): C_KG ≥ τ_high. Ràng buộc số ứng viên là phần of-record chưa có trong bài.
        if (kg_conf >= self.cfg.KG_HIGH_CONF_THRESHOLD
                and len(kg_ranked) >= self.cfg.KG_HIGH_CONF_MIN_CANDIDATES):
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
            ranked = sorted(rerank.items(), key=lambda x: x[1], reverse=True)[:self.cfg.TOP_K]
            return [d for d, _ in ranked], "kg_high_conf"

        # BRANCH 3: DISAGREE (KG-anchored)
        pool, seen = [], set()
        for d in kg_ranked[:10]:
            if d not in seen:
                pool.append(d); seen.add(d)
        for d in rag_ranked[:3]:
            if d not in seen:
                pool.append(d); seen.add(d)
        bge_scores = self.bge_score_candidates(query, pool)
        rag_top3 = set(rag_ranked[:3])
        kg_top3 = set(kg_ranked[:3])
        rerank = {}
        for d in pool:
            bge_norm = _sigmoid(bge_scores.get(d, 0.0)) if bge_scores else 0.0
            kg_s = kg_n.get(d, 0.0); rag_s = rag_n.get(d, 0.0)
            s = 0.40 * bge_norm + 0.40 * kg_s + 0.20 * rag_s
            if d in kg_top3 and d in rag_top3:
                s += 0.30
            elif d in kg_top3:
                s += 0.10
            elif d in rag_top3 and d not in kg_ranked[:5]:
                s -= 0.05
            if kg_conf > self.cfg.KG_MED_CONF_THRESHOLD and d == kg_top1:
                s += 0.15
            rerank[d] = s
        ranked = sorted(rerank.items(), key=lambda x: x[1], reverse=True)[:self.cfg.TOP_K]
        return [d for d, _ in ranked], "disagree_kg_anchored"

    # ============================================================
    #  TOP-LEVEL: chạy 3 hệ thống cho 1 query
    # ============================================================
    def run_all(self, query: str) -> Dict:
        t0 = time.perf_counter()
        symptoms = self.extract_symptoms(query)
        if not symptoms:
            symptoms = [query]
        t_extract = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        vectors = self.embed_symptoms(symptoms)
        t_embed = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        rag_base, rag_scores = self.retrieve_rag(query, vectors, symptoms)
        t_rag = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        kg_base, kg_subgraphs, kg_scores, kg_max = self.retrieve_kg(vectors, symptoms)
        t_kg = (time.perf_counter() - t3) * 1000

        t4 = time.perf_counter()
        hybrid_base, fusion_mode = self.hybrid_fusion_v11(
            query, rag_base, rag_scores, kg_base, kg_scores, kg_max
        )
        t_hybrid = (time.perf_counter() - t4) * 1000

        # BGE rerank cho RAG/KG baseline (giống notebook)
        rag_bge, rag_bge_s = self.bge_rerank(query, rag_base)
        kg_bge, kg_bge_s = self.bge_rerank(query, kg_base)
        rag_top1 = self.llm_cross_rerank(symptoms, rag_bge, kg_subgraphs)
        kg_top1 = self.llm_cross_rerank(symptoms, kg_bge, kg_subgraphs)

        # Hybrid: BGE chứ KHÔNG dùng LLM rerank (theo notebook v11)
        hybrid_bge, _ = self.bge_rerank(query, hybrid_base)
        hybrid_top1 = [hybrid_bge[0]] if hybrid_bge else []

        def _assemble(top1, base):
            seen, final = set(top1), list(top1)
            for d in base:
                if d not in seen:
                    final.append(d); seen.add(d)
            return final

        return {
            "symptoms": symptoms,
            "rag": (_assemble(rag_top1, rag_base), rag_scores, rag_base),
            "kg":  (_assemble(kg_top1, kg_base), kg_scores, kg_base, kg_subgraphs, kg_max),
            "hybrid": (_assemble(hybrid_top1, hybrid_base), fusion_mode, hybrid_base),
            "rag_top1": rag_top1[0] if rag_top1 else "",
            "kg_top1": kg_top1[0] if kg_top1 else "",
            "hybrid_top1": hybrid_top1[0] if hybrid_top1 else "",
            "latency_ms": {
                "extract": round(t_extract, 1),
                "embed": round(t_embed, 1),
                "rag": round(t_rag, 1),
                "kg": round(t_kg, 1),
                "hybrid": round(t_hybrid, 1),
                "total": round(t_extract + t_embed + t_rag + t_kg + t_hybrid, 1),
            },
        }

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

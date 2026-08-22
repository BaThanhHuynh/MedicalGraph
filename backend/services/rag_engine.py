import os
import re
import json
import logging
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Generator
import numpy as np
import pandas as pd

from backend.config import config

logger = logging.getLogger("rag_engine")
logging.basicConfig(level=logging.INFO)

# Import MedKG-HRR Engine chuẩn bài báo v16
from backend.services.medkghrr_engine import medkghrr_engine

class SegmentedEncoder:
    """Wrapper mã hóa văn bản tiếng Việt sử dụng pyvi và PhoBERT contrastive."""
    def __init__(self, raw_model, tokenizer_fn):
        self.model = raw_model
        self.tok = tokenizer_fn

    def encode(self, texts, batch_size=32, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        segmented = [self.tok.tokenize(str(t)) for t in texts]
        return self.model.encode(segmented, batch_size=batch_size, **kwargs)

class MedicalRAGEngine:
    """
    MedKG-HRR Engine (Chuẩn theo Bài báo v16):
    
    Pipeline MedKG-HRR đầy đủ (delegate sang medkghrr_engine.py):
    - Trích xuất triệu chứng: Dict matching (PhoBERT cosine)
    - Kênh KG: Neo4j Cypher (4 tín hiệu)
    - Kênh Dense: FAISS PhoBERT contrastive (per-symptom + query)
    - Kênh Sparse: BM25 Okapi (expanded query)
    - RRF Fusion k=60
    - BGE-reranker-v2-m3 (cross-encoder)
    - Hybrid Fusion v11: Consensus / KG High Conf / Disagree
    
    Phần UI bổ trợ (KHÔNG thuộc pipeline MedKG-HRR theo bài báo):
    - Gemini LLM: sinh văn bản chẩn đoán hỗ trợ (bài báo nêu rõ hệ "không sinh văn bản")
    """

    def __init__(self):
        self.device = "cpu"
        self.embed_model = None
        self.raw_embed = None
        self.faiss_index = None
        self.df_rag = None
        self.corpus = []
        self.bm25 = None
        self.df_medical = None
        self.disease_dict = {}  # Tra cứu nhanh ICD-10, chuyên khoa từ 1.109 bệnh
        self.is_ready = False
        self.last_load_error = None

    def initialize(self):
        """Khởi tạo MedKG-HRR pipeline chuẩn bài báo v16."""
        try:
            logger.info("=" * 60)
            logger.info("   KHỞI TẠO HỆ THỐNG MEDKG-HRR (CHUẨN BÀI BÁO V16)")
            logger.info("=" * 60)
            
            # 1. Khởi tạo MedKG-HRR Engine đầy đủ (3 kênh + 3 nhánh)
            medkghrr_engine.initialize()
            
            # 2. Lấy tham chiếu từ engine mới cho backward compatibility
            if medkghrr_engine.is_ready:
                self.embed_model = medkghrr_engine._embed_model
                self.raw_embed = medkghrr_engine._raw_embed
                self.faiss_index = medkghrr_engine._faiss_index
                self.df_rag = medkghrr_engine._df_rag
                self.bm25 = medkghrr_engine._bm25
                self.disease_dict = medkghrr_engine._disease_dict
                self.df_medical = medkghrr_engine._df_medical
                
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Corpus cho Gemini context
                if medkghrr_engine._corpus:
                    self.corpus = medkghrr_engine._corpus

            self.is_ready = medkghrr_engine.is_ready
            if self.is_ready:
                logger.info("=" * 60)
                logger.info(" [✓] MEDKG-HRR BACKEND ĐÃ SẴN SÀNG (PIPELINE V16)")
                logger.info("=" * 60)
            else:
                logger.warning("MedKG-HRR engine chưa sẵn sàng — kiểm tra log.")
                
        except Exception as e:
            self.is_ready = False
            self.last_load_error = str(e)
            logger.error(f"[X] Lỗi khởi tạo MedKG-HRR: {e}", exc_info=True)

    def _load_medical_database(self):
        med_path = config.CSV_MEDICAL
        if not os.path.exists(med_path):
            logger.warning(f"Không tìm thấy file bệnh chuẩn tại: {med_path}")
            return

        logger.info(f"[*] Đang nạp danh mục bệnh chuẩn y khoa từ: {med_path}")
        try:
            self.df_medical = pd.read_csv(med_path, delimiter=";", on_bad_lines="skip", engine="python", encoding="utf-8")
        except Exception as e:
            logger.warning(f"Lỗi đọc {med_path}: {e}")
            return

        # Chuẩn hóa từ điển tra cứu bệnh học 1.109 bệnh neo ICD-10
        self.disease_dict = {}
        for _, row in self.df_medical.iterrows():
            disease_name = str(row.get("disease_name", row.get("ten_benh", ""))).strip()
            if not disease_name or disease_name.lower() == "nan":
                continue
            
            icd10 = str(row.get("icd10_code", row.get("ma_icd10", ""))).strip()
            if icd10.lower() == "nan": icd10 = ""

            icd_name = str(row.get("icd10_name", "")).strip()
            if icd_name.lower() == "nan": icd_name = ""
            
            specialty = str(row.get("chuyen_khoa_goc", row.get("chuyen_khoa", "Đa khoa / Nội khoa"))).strip()
            if specialty.lower() == "nan": specialty = "Đa khoa / Nội khoa"

            symptoms = str(row.get("positive_symptoms", row.get("positive_symptoms_norm", ""))).strip()
            if symptoms.lower() == "nan": symptoms = ""

            chan_doan = str(row.get("chan_doan_text", "")).strip()
            if chan_doan.lower() == "nan": chan_doan = ""

            self.disease_dict[disease_name.lower()] = {
                "name": disease_name,
                "icd10": icd10,
                "icd10_name": icd_name,
                "specialty": specialty,
                "symptoms": symptoms,
                "chan_doan": chan_doan
            }
        logger.info(f"[✓] Đã lập chỉ mục {len(self.disease_dict)} bệnh lý chuẩn theo ICD-10.")

    # ============================================================
    # MEDKG-HRR PIPELINE (CHUẨN BÀI BÁO V16)
    # Delegate sang medkghrr_engine — đầy đủ 3 kênh + 3 nhánh
    # ============================================================
    def retrieve_hybrid_rrf(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Pipeline MedKG-HRR chuẩn bài báo v16 (Mục 4):
        1. Trích xuất triệu chứng (Dict matching)
        2. FAISS per-symptom + BM25 expanded → RRF k=60
        3. KG Cypher (4 tín hiệu Neo4j)
        4. Hybrid Fusion v11 (Consensus / KG High Conf / Disagree)
        5. BGE-reranker-v2-m3 tái xếp hạng
        """
        if not self.is_ready:
            return []

        # Chạy pipeline đầy đủ
        pipeline_result = medkghrr_engine.run_pipeline(query)
        
        if "error" in pipeline_result:
            logger.warning(f"Pipeline error: {pipeline_result['error']}")
            return []

        # Lưu pipeline metadata để process_query/process_query_stream sử dụng
        self._last_pipeline_result = pipeline_result

        # Format kết quả cho backward compatibility với frontend
        ranked = pipeline_result.get("ranked_diseases", [])
        results = []
        max_score = ranked[0]["score"] if ranked and ranked[0].get("score", 0) > 0 else 1.0

        for item in ranked[:top_k]:
            score = item.get("score", 0.0)
            rank = item.get("rank", 0)
            # Tính tỷ lệ tương đồng chuẩn hóa (%)
            sim_pct = round(min(98.5, max(60.0,
                (score / max_score * 95.0 if max_score > 0 else 70.0) - (rank - 1) * 3.5
            )), 1) if score > 0 else 70.0

            results.append({
                "rank": rank,
                "disease": item.get("disease", ""),
                "symptom": item.get("symptom", "Xem chi tiết trong hồ sơ bệnh án"),
                "icd10": item.get("icd10", ""),
                "specialty": item.get("specialty", "Đa khoa / Nội khoa"),
                "score": float(score),
                "similarity_pct": sim_pct,
                "chunk_evidence": "",  # Sẽ bổ sung từ corpus nếu cần
                # Thông tin pipeline mới (bài báo v16)
                "kg_score": item.get("kg_score", 0.0),
                "kg_evidence": item.get("kg_evidence", []),
                "source": item.get("source", ""),
            })

        return results

    # ============================================================
    # GEMINI LLM MEDICAL CONSULTANT (SYNC & STREAMING)
    # ============================================================
    def call_gemini(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        api_key = config.GEMINI_API_KEY
        if not api_key:
            return (
                "⚠️ **Chưa cấu hình Gemini API Key!**\n\n"
                "Vui lòng nhấn vào biểu tượng **Flash/Cài đặt** ở góc trên để nhập Gemini API Key hoặc cập nhật file `.env`."
            )

        model_name = config.GEMINI_MODEL
        temperature = config.TEMPERATURE

        # 1. Gọi qua SDK
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            config_params = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2048,
            )
            if system_instruction:
                config_params.system_instruction = system_instruction

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config_params
            )
            if response and response.text:
                return response.text.strip()
        except Exception as sdk_err:
            logger.warning(f"Gọi Gemini SDK thất bại ({sdk_err}), chuyển sang REST API fallback...")

        # 2. Fallback REST API
        import urllib.request
        import urllib.error

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 2048
            }
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                return "Không nhận được phản hồi hợp lệ từ Gemini API."
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8")
            logger.error(f"HTTP Error từ Gemini: {http_err.code} - {err_body}")
            return f"❌ **Lỗi gọi Gemini API ({http_err.code})**: {err_body}"
        except Exception as e:
            logger.error(f"Lỗi kết nối Gemini REST: {e}")
            return f"❌ **Lỗi kết nối tới Gemini API**: {str(e)}"

    def stream_gemini(self, prompt: str, system_instruction: Optional[str] = None) -> Generator[str, None, None]:
        """Stream các token văn bản sinh từ Gemini LLM với độ trễ cực thấp."""
        api_key = config.GEMINI_API_KEY
        if not api_key:
            yield "⚠️ **Chưa cấu hình Gemini API Key!**\n\nVui lòng nhấn vào biểu tượng **Flash/Cài đặt** để nhập Gemini API Key hoặc cấu hình file `.env`."
            return

        model_name = config.GEMINI_MODEL
        temperature = config.TEMPERATURE

        # 1. Thử gọi Google GenAI SDK Stream
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            config_params = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2048,
            )
            if system_instruction:
                config_params.system_instruction = system_instruction

            stream = client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
                config=config_params
            )
            for chunk in stream:
                if chunk and chunk.text:
                    yield chunk.text
            return
        except Exception as sdk_err:
            logger.warning(f"Gọi Gemini SDK stream thất bại ({sdk_err}), chuyển sang REST streaming fallback...")

        # 2. Fallback REST SSE Stream
        import urllib.request
        import urllib.error

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse&key={api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 2048
            }
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                for line in resp:
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data: "):
                        data_str = decoded[6:].strip()
                        if not data_str:
                            continue
                        try:
                            chunk_data = json.loads(data_str)
                            candidates = chunk_data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for p in parts:
                                    t = p.get("text", "")
                                    if t:
                                        yield t
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"Lỗi REST stream: {e}")
            yield f"\n\n❌ **Lỗi kết nối tới Gemini API**: {str(e)}"

    def classify_intent(self, user_query: str) -> str:
        lower_q = user_query.strip().lower()
        greetings = ["xin chào", "chào", "hello", "hi", "hey", "cảm ơn", "thank you", "thanks", "bạn là ai", "bạn tên gì", "tạm biệt", "bye"]
        if any(lower_q == g or lower_q.startswith(g + " ") for g in greetings) and len(lower_q.split()) <= 6:
            return "chat"
        return "diagnose"

    def process_query(self, user_query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Quy trình MedKG-HRR + Gemini tư vấn (Đồng bộ):
        1. Phân loại intent
        2. MedKG-HRR Pipeline chuẩn bài báo v16 (3 kênh + 3 nhánh)
        3. Tổng hợp chẩn đoán & tư vấn qua Gemini LLM (phần UI bổ trợ)
        """
        start_time = time.time()
        intent = self.classify_intent(user_query)

        if intent == "chat":
            chat_prompt = f"Người dùng nhắn: \"{user_query}\"\nHãy trả lời một cách lịch sự, chuyên nghiệp với vai trò là Trợ lý AI Y tế MedKG-AI, sẵn sàng phân tích triệu chứng hoặc giải đáp thắc mắc y khoa dựa trên Đồ thị tri thức y tế ICD-10."
            response_text = self.call_gemini(
                prompt=chat_prompt,
                system_instruction="Bạn là Bác sĩ Trợ lý AI MedKG-AI thông minh, thân thiện, chu đáo và chuẩn xác."
            )
            return {
                "intent": "chat",
                "answer": response_text,
                "matched_diseases": [],
                "model_used": config.GEMINI_MODEL,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "disclaimer": None
            }

        # Thực hiện MedKG-HRR Pipeline chuẩn bài báo v16
        top_k = top_k or config.TOP_K
        matched_diseases = self.retrieve_hybrid_rrf(user_query, top_k=top_k)

        # Lấy pipeline metadata (triệu chứng, fusion mode, KG evidence)
        pipeline_meta = getattr(self, '_last_pipeline_result', {})
        symptoms_extracted = pipeline_meta.get("symptoms_extracted", [])
        fusion_mode = pipeline_meta.get("fusion_mode", "unknown")
        pipeline_latency = pipeline_meta.get("latency_ms", {})

        # Xây dựng ngữ cảnh y khoa — bao gồm KG evidence nếu có
        context_blocks = []
        for item in matched_diseases:
            block = f"### [BỆNH]: {item['disease']}"
            if item.get("icd10"):
                block += f" (Mã ICD-10: {item['icd10']})"
            block += f"\n- Chuyên khoa khám đề xuất: {item['specialty']}"
            if item.get("symptom"):
                block += f"\n- Triệu chứng đặc trưng: {item['symptom']}"
            # KG evidence từ đồ thị tri thức
            kg_ev = item.get("kg_evidence", [])
            if kg_ev:
                block += f"\n- Bằng chứng đồ thị KG: {', '.join(kg_ev[:5])}"
            if item.get("chunk_evidence"):
                block += f"\n- Hồ sơ y khoa trích xuất: {item['chunk_evidence'][:250]}..."
            if item.get("source"):
                block += f"\n- Nguồn: {item['source']}"
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks) if context_blocks else "Không tìm thấy hồ sơ bệnh lý khớp trong cơ sở dữ liệu."

        # Thêm thông tin triệu chứng đã trích
        if symptoms_extracted:
            context_str = f"[TRIỆU CHỨNG ĐÃ TRÍCH XUẤT]: {', '.join(symptoms_extracted)}\n[CHẾ ĐỘ PHỐI HỢP]: {fusion_mode}\n\n{context_str}"

        system_instruction = (
            "Bạn là Bác sĩ Chuyên gia Trợ lý AI Y tế thuộc hệ thống MedKG-AI (Khung xếp hạng MedKG-HRR). "
            "Nhiệm vụ của bạn là phân tích triệu chứng người dùng dựa trên [NGỮ CẢNH Y KHOA ĐÃ TRUY XUẤT TỪ CƠ SỞ DỮ LIỆU ĐỒ THỊ TRI THỨC VÀ ICD-10].\n"
            "YÊU CẦU LÂM SÀNG:\n"
            "1. Phân tích nguyên nhân & bệnh lý nghi ngờ hàng đầu (nêu rõ mã ICD-10 và tên bệnh theo ngữ cảnh).\n"
            "2. So sánh và đối chiếu các triệu chứng bệnh nhân mô tả với triệu chứng điển hình trong tài liệu y khoa.\n"
            "3. Đề xuất bước xử trí: Chuyên khoa cần khám, xét nghiệm cận lâm sàng cần làm hoặc chăm sóc ban đầu.\n"
            "4. Dấu hiệu cảnh báo nguy hiểm (Red Flags) cần nhập viện khẩn cấp.\n"
            "5. Trình bày bằng Markdown khoa học, gạch đầu dòng rõ ràng, mạch lạc, chính xác và đồng cảm với người bệnh."
        )

        user_prompt = f"""[NGỮ CẢNH TRI THỨC Y KHOA TRUY XUẤT TỪ MEDKG-HRR (PHOBERT + FAISS + BM25 + NEO4J KG + ICD-10)]:
{context_str}

[MÔ TẢ TRIỆU CHỨNG / CÂU HỎI BỆNH NHÂN]:
"{user_query}"

Hãy đóng vai Bác sĩ Trợ lý Y khoa phân tích toàn diện, đưa ra các chẩn đoán phân biệt tiềm năng nhất và hướng dẫn bệnh nhân:"""

        ai_response = self.call_gemini(prompt=user_prompt, system_instruction=system_instruction)

        return {
            "intent": "diagnose",
            "answer": ai_response,
            "matched_diseases": matched_diseases,
            "model_used": config.GEMINI_MODEL,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "disclaimer": "⚠️ Lưu ý: Kết quả phân tích từ AI chỉ mang tính chất tham khảo học thuật và định hướng thông tin y tế theo hệ thống MedKG-HRR, không thay thế cho chẩn đoán hoặc chỉ định trực tiếp từ Bác sĩ chuyên khoa.",
            # Pipeline metadata mới (bài báo v16)
            "symptoms_extracted": symptoms_extracted,
            "fusion_mode": fusion_mode,
            "pipeline_latency": pipeline_latency,
        }

    def process_query_stream(self, user_query: str, top_k: Optional[int] = None) -> Generator[str, None, None]:
        """
        Quy trình RAG Y tế chuẩn MedKG-HRR (Stream Server-Sent Events):
        - Bắn ngay metadata (intent, danh sách bệnh khớp, model) -> time-to-first-token tức thì
        - Stream từng token của Gemini theo dạng typewriter
        - Bắn sự kiện done kèm latency
        """
        start_time = time.time()
        intent = self.classify_intent(user_query)

        if intent == "chat":
            chat_prompt = f"Người dùng nhắn: \"{user_query}\"\nHãy trả lời một cách lịch sự, chuyên nghiệp với vai trò là Trợ lý AI Y tế MedKG-AI, sẵn sàng phân tích triệu chứng hoặc giải đáp thắc mắc y khoa dựa trên Đồ thị tri thức y tế ICD-10."
            
            yield json.dumps({
                "type": "meta",
                "intent": "chat",
                "matched_diseases": [],
                "model_used": config.GEMINI_MODEL
            }) + "\n"

            for token in self.stream_gemini(
                prompt=chat_prompt,
                system_instruction="Bạn là Bác sĩ Trợ lý AI MedKG-AI thông minh, thân thiện, chu đáo và chuẩn xác."
            ):
                yield json.dumps({"type": "token", "content": token}) + "\n"

            latency_ms = round((time.time() - start_time) * 1000, 2)
            yield json.dumps({"type": "done", "latency_ms": latency_ms}) + "\n"
            return

        # Thực hiện MedKG-HRR Pipeline chuẩn bài báo v16
        top_k = top_k or config.TOP_K
        matched_diseases = self.retrieve_hybrid_rrf(user_query, top_k=top_k)

        # Lấy pipeline metadata
        pipeline_meta = getattr(self, '_last_pipeline_result', {})
        symptoms_extracted = pipeline_meta.get("symptoms_extracted", [])
        fusion_mode = pipeline_meta.get("fusion_mode", "unknown")

        # Bắn ngay metadata cho frontend render tức thì
        yield json.dumps({
            "type": "meta",
            "intent": "diagnose",
            "matched_diseases": matched_diseases,
            "model_used": config.GEMINI_MODEL,
            "symptoms_extracted": symptoms_extracted,
            "fusion_mode": fusion_mode,
            "disclaimer": "⚠️ Lưu ý: Kết quả phân tích từ AI chỉ mang tính chất tham khảo học thuật và định hướng thông tin y tế theo hệ thống MedKG-HRR, không thay thế cho chẩn đoán hoặc chỉ định trực tiếp từ Bác sĩ chuyên khoa."
        }) + "\n"

        context_blocks = []
        for item in matched_diseases:
            block = f"### [BỆNH]: {item['disease']}"
            if item.get("icd10"):
                block += f" (Mã ICD-10: {item['icd10']})"
            block += f"\n- Chuyên khoa khám đề xuất: {item['specialty']}"
            if item.get("symptom"):
                block += f"\n- Triệu chứng đặc trưng: {item['symptom']}"
            kg_ev = item.get("kg_evidence", [])
            if kg_ev:
                block += f"\n- Bằng chứng đồ thị KG: {', '.join(kg_ev[:5])}"
            if item.get("chunk_evidence"):
                block += f"\n- Hồ sơ y khoa trích xuất: {item['chunk_evidence'][:250]}..."
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks) if context_blocks else "Không tìm thấy hồ sơ bệnh lý khớp trong cơ sở dữ liệu."

        if symptoms_extracted:
            context_str = f"[TRIỆU CHỨNG ĐÃ TRÍCH XUẤT]: {', '.join(symptoms_extracted)}\n[CHẾ ĐỘ PHỐI HỢP]: {fusion_mode}\n\n{context_str}"

        system_instruction = (
            "Bạn là Bác sĩ Chuyên gia Trợ lý AI Y tế thuộc hệ thống MedKG-AI (Khung xếp hạng MedKG-HRR). "
            "Nhiệm vụ của bạn là phân tích triệu chứng người dùng dựa trên [NGỮ CẢNH Y KHOA ĐÃ TRUY XUẤT TỪ CƠ SỞ DỮ LIỆU ĐỒ THỊ TRI THỨC VÀ ICD-10].\n"
            "YÊU CẦU LÂM SÀNG:\n"
            "1. Phân tích nguyên nhân & bệnh lý nghi ngờ hàng đầu (nêu rõ mã ICD-10 và tên bệnh theo ngữ cảnh).\n"
            "2. So sánh và đối chiếu các triệu chứng bệnh nhân mô tả với triệu chứng điển hình trong tài liệu y khoa.\n"
            "3. Đề xuất bước xử trí: Chuyên khoa cần khám, xét nghiệm cận lâm sàng cần làm hoặc chăm sóc ban đầu.\n"
            "4. Dấu hiệu cảnh báo nguy hiểm (Red Flags) cần nhập viện khẩn cấp.\n"
            "5. Trình bày bằng Markdown khoa học, gạch đầu dòng rõ ràng, mạch lạc, chính xác và đồng cảm với người bệnh."
        )

        user_prompt = f"""[NGỮ CẢNH TRI THỨC Y KHOA TRUY XUẤT TỪ MEDKG-HRR (PHOBERT + FAISS + BM25 + NEO4J KG + ICD-10)]:
{context_str}

[MÔ TẢ TRIỆU CHỨNG / CÂU HỎI BỆNH NHÂN]:
"{user_query}"

Hãy đóng vai Bác sĩ Trợ lý Y khoa phân tích toàn diện, đưa ra các chẩn đoán phân biệt tiềm năng nhất và hướng dẫn bệnh nhân:"""

        for token in self.stream_gemini(prompt=user_prompt, system_instruction=system_instruction):
            yield json.dumps({"type": "token", "content": token}) + "\n"

        latency_ms = round((time.time() - start_time) * 1000, 2)
        yield json.dumps({"type": "done", "latency_ms": latency_ms}) + "\n"

# Global singleton engine
rag_engine = MedicalRAGEngine()

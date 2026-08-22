import logging
import io
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from gtts import gTTS

from backend.config import config
from backend.services.rag_engine import rag_engine

logger = logging.getLogger("api_chat")
router = APIRouter(prefix="/api", tags=["Medical Chat & RAG"])

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Mô tả triệu chứng hoặc câu hỏi của bệnh nhân")
    gemini_api_key: Optional[str] = Field(None, description="API Key truyền từ client (nếu có)")
    gemini_model: Optional[str] = Field(None, description="Mô hình LLM tùy chọn")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Số lượng bệnh tương đồng cần truy xuất")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="Độ sáng tạo")

class DiseaseMatch(BaseModel):
    rank: int
    disease: str
    symptom: str
    icd10: Optional[str] = ""
    specialty: Optional[str] = "Đa khoa"
    score: float
    similarity_pct: float
    chunk_evidence: Optional[str] = ""
    kg_score: Optional[float] = 0.0
    kg_evidence: Optional[List[str]] = []
    source: Optional[str] = ""

class ChatResponse(BaseModel):
    intent: str
    matched_diseases: List[DiseaseMatch]
    answer: str
    disclaimer: Optional[str] = None
    model_used: str
    latency_ms: float
    symptoms_extracted: Optional[List[str]] = []
    fusion_mode: Optional[str] = "unknown"
    pipeline_latency: Optional[Dict[str, float]] = {}

@router.post("/chat", response_model=ChatResponse)
async def chat_with_medical_rag(payload: ChatRequest):
    """
    Endpoint chính xử lý tương tác tư vấn y tế RAG:
    1. Nhận triệu chứng/câu hỏi từ người dùng.
    2. Truy xuất Top-K bệnh tương đồng từ đồ thị tri thức ICD-10 (FAISS + BM25).
    3. Sinh câu trả lời lâm sàng chi tiết qua Google Gemini (GenAI).
    """
    user_query = payload.message.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Nội dung câu hỏi không được để trống.")

    # Override config tạm thời nếu client truyền vào
    if payload.gemini_api_key:
        config.GEMINI_API_KEY = payload.gemini_api_key.strip()
    if payload.gemini_model:
        config.GEMINI_MODEL = payload.gemini_model.strip()
    if payload.temperature is not None:
        config.TEMPERATURE = payload.temperature

    try:
        result = rag_engine.process_query(user_query, top_k=payload.top_k)
        return result
    except Exception as e:
        logger.error(f"Lỗi khi xử lý truy vấn RAG: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")

@router.post("/chat/stream")
async def chat_with_medical_rag_stream(payload: ChatRequest):
    """
    Endpoint Streaming phản hồi chẩn đoán y tế thời gian thực (Server-Sent Events).
    - Giảm thiểu độ trễ phản hồi ban đầu (Time to first token < 300ms).
    - Hiệu ứng typewriter mượt mà khi LLM sinh câu trả lời.
    """
    user_query = payload.message.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Nội dung câu hỏi không được để trống.")

    if payload.gemini_api_key:
        config.GEMINI_API_KEY = payload.gemini_api_key.strip()
    if payload.gemini_model:
        config.GEMINI_MODEL = payload.gemini_model.strip()
    if payload.temperature is not None:
        config.TEMPERATURE = payload.temperature

    def event_stream():
        try:
            # Gửi ngay sự kiện khởi tạo kết nối để browser không bị timeout/network error
            import json
            yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            for item in rag_engine.process_query_stream(user_query, top_k=payload.top_k):
                yield f"data: {item.strip()}\n\n"
        except Exception as e:
            logger.error(f"Lỗi trong stream event: {e}", exc_info=True)
            import json
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/tts")
async def text_to_speech(text: str = Query(..., description="Văn bản tiếng Việt cần đọc")):
    """
    Endpoint Text-To-Speech (TTS) chuẩn Tiếng Việt 100%:
    - Sử dụng mô hình Google Text-to-Speech (gTTS) ngôn ngữ tiếng Việt (vi).
    - Trả về tệp âm thanh MP3 trực tiếp cho trình duyệt phát mượt mà không phụ thuộc vào hệ điều hành.
    """
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Thiếu nội dung văn bản.")

    try:
        # Giới hạn 1500 ký tự đầu tiên để phản hồi cực nhanh
        short_text = clean_text[:1500]
        fp = io.BytesIO()
        tts = gTTS(text=short_text, lang="vi", slow=False)
        tts.write_to_fp(fp)
        fp.seek(0)
        return Response(content=fp.read(), media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"Lỗi sinh âm thanh TTS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi sinh âm thanh TTS: {str(e)}")

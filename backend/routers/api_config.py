import os
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from backend.config import config
from backend.services.rag_engine import rag_engine

router = APIRouter(prefix="/api", tags=["Configuration & Management"])

class ConfigUpdateRequest(BaseModel):
    gemini_api_key: Optional[str] = Field(None, description="Google Gemini API Key")
    gemini_model: Optional[str] = Field(None, description="Tên mô hình Gemini (VD: gemini-2.5-flash)")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="Độ sáng tạo / ngẫu nhiên của LLM")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Số lượng bệnh lý tương đồng cần truy xuất")
    phobert_model_path: Optional[str] = Field(None, description="Đường dẫn model PhoBERT")

class TestLLMRequest(BaseModel):
    api_key: Optional[str] = None
    model_name: Optional[str] = None

@router.get("/status")
async def get_system_status():
    """Lấy thông tin trạng thái hoạt động của Backend, PhoBERT Model, FAISS và Gemini LLM."""
    phobert_loaded = rag_engine.is_ready and rag_engine.embed_model is not None
    faiss_loaded = rag_engine.is_ready and rag_engine.faiss_index is not None
    total_vectors = rag_engine.faiss_index.ntotal if faiss_loaded else 0
    total_chunks = len(rag_engine.corpus) if (rag_engine.corpus is not None) else 0
    total_diseases = len(rag_engine.disease_dict) if (rag_engine.disease_dict is not None) else 0

    return {
        "status": "online",
        "version": "2.0.0 (MedKG-HRR)",
        "framework": "MedKG-HRR: Hybrid Retrieval (FAISS Dense + BM25 Sparse + RRF k=60) + Neural Reranking",
        "phobert": {
            "loaded": phobert_loaded,
            "model_path": config.PHOBERT_MODEL_PATH,
            "device": str(getattr(rag_engine, "device", "cpu"))
        },
        "faiss": {
            "loaded": faiss_loaded,
            "total_vectors": total_vectors,
            "total_chunks": total_chunks,
            "total_canonical_diseases": total_diseases,
            "index_path": config.FAISS_INDEX_PATH
        },
        "llm": {
            "provider": "Google Gemini",
            "api_key_configured": bool(config.GEMINI_API_KEY),
            "current_model": config.GEMINI_MODEL,
            "temperature": config.TEMPERATURE,
            "top_k": config.TOP_K,
            "available_models": [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ]
        },
        "initialization_error": rag_engine.last_load_error
    }

@router.get("/config")
async def get_config():
    """Lấy cấu hình hiện tại của hệ thống."""
    return config.to_dict()

@router.post("/config")
async def update_config(payload: ConfigUpdateRequest):
    """Cập nhật cấu hình runtime cho hệ thống (API Key, Model, Temperature, Top-K)."""
    updates = {}
    if payload.gemini_api_key is not None:
        updates["GEMINI_API_KEY"] = payload.gemini_api_key.strip()
    if payload.gemini_model is not None:
        updates["GEMINI_MODEL"] = payload.gemini_model.strip()
    if payload.temperature is not None:
        updates["TEMPERATURE"] = payload.temperature
    if payload.top_k is not None:
        updates["TOP_K"] = payload.top_k
    if payload.phobert_model_path is not None:
        updates["PHOBERT_MODEL_PATH"] = payload.phobert_model_path.strip()

    config.update(**updates)
    
    # Nếu đổi đường dẫn model, thử nạp lại engine
    if payload.phobert_model_path is not None:
        rag_engine.initialize()

    return {
        "success": True,
        "message": "Cấu hình hệ thống đã được cập nhật thành công!",
        "config": config.to_dict()
    }

@router.post("/test-llm")
async def test_llm_connection(payload: TestLLMRequest):
    """Kiểm tra kết nối tới Gemini API với API Key và Model được cung cấp."""
    test_key = payload.api_key.strip() if payload.api_key else config.GEMINI_API_KEY
    if not test_key:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp Gemini API Key để kiểm tra.")

    model_name = payload.model_name.strip() if payload.model_name else config.GEMINI_MODEL
    
    # Gọi thử một câu ngắn
    prompt = "Trả lời ngắn gọn đúng 1 từ: 'OK' nếu bạn nhận được tin nhắn này."
    
    # Tạm thời gán key để test
    original_key = config.GEMINI_API_KEY
    original_model = config.GEMINI_MODEL
    try:
        config.GEMINI_API_KEY = test_key
        config.GEMINI_MODEL = model_name
        result = rag_engine.call_gemini(prompt)
        
        if "Lỗi" in result or "Error" in result:
            return {"success": False, "error": result}
            
        return {
            "success": True,
            "message": f"Kết nối Gemini API ({model_name}) thành công!",
            "response": result
        }
    finally:
        config.GEMINI_API_KEY = original_key
        config.GEMINI_MODEL = original_model

@router.post("/reload-engine")
async def reload_engine():
    """Tải lại PhoBERT Model và FAISS Index."""
    rag_engine.initialize()
    return {
        "success": rag_engine.is_ready,
        "message": "Đã thực hiện nạp lại RAG Engine!",
        "error": rag_engine.last_load_error
    }

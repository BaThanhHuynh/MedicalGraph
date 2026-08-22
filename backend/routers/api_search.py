from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.rag_engine import rag_engine
from backend.services.medkghrr_engine import medkghrr_engine

router = APIRouter(prefix="/api", tags=["Medical Search"])

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Mô tả triệu chứng hoặc từ khóa bệnh cần tra cứu")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Số kết quả cần lấy")

@router.post("/search")
async def search_medical_database(payload: SearchRequest):
    """
    Tra cứu trực tiếp qua MedKG-HRR Pipeline chuẩn bài báo v16:
    1. Trích xuất triệu chứng (Dict matching PhoBERT)
    2. FAISS per-symptom + BM25 expanded → RRF k=60
    3. KG Cypher (Neo4j 4 tín hiệu)
    4. Hybrid Fusion v11 (Consensus / KG High Conf / Disagree)
    5. BGE-reranker-v2-m3
    """
    if not rag_engine.is_ready:
        raise HTTPException(
            status_code=503, 
            detail="Hệ thống MedKG-HRR chưa sẵn sàng. Vui lòng kiểm tra lại PhoBERT model và FAISS index."
        )
        
    query = payload.query.strip()
    results = rag_engine.retrieve_hybrid_rrf(query, top_k=payload.top_k)
    
    # Lấy pipeline metadata
    pipeline_meta = getattr(rag_engine, '_last_pipeline_result', {})
    
    return {
        "query": query,
        "framework": "MedKG-HRR v16 (3 kênh + 3 nhánh)",
        "symptoms_extracted": pipeline_meta.get("symptoms_extracted", []),
        "fusion_mode": pipeline_meta.get("fusion_mode", "unknown"),
        "total_results": len(results),
        "results": results,
        "pipeline_latency": pipeline_meta.get("latency_ms", {}),
    }

@router.get("/pipeline-info")
async def get_pipeline_info():
    """
    Trạng thái pipeline MedKG-HRR — components nào đang active,
    kiến trúc, và cấu hình hiện tại.
    """
    return medkghrr_engine.get_pipeline_status()


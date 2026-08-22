import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from backend.config import config
from backend.services.rag_engine import rag_engine
from backend.routers import api_config, api_chat, api_search

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("medical_backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời khởi động và tắt của ứng dụng FastAPI."""
    logger.info("=========================================================")
    logger.info("   MEDKG-AI BACKEND ĐANG KHỞI ĐỘNG (PhoBERT + GEMINI RAG)")
    logger.info("=========================================================")
    
    # Khởi tạo RAG Engine
    try:
        rag_engine.initialize()
    except Exception as e:
        logger.error(f"Khởi tạo RAG Engine thất bại khi startup: {e}")
        
    yield
    
    logger.info("Backend đang dừng hoạt động...")

app = FastAPI(
    title="MedKG-AI: Hệ Thống Trợ Lý Y Khoa (PhoBERT + Gemini RAG)",
    description="Backend API quản lý PhoBERT local contrastive embedding, FAISS vector search và Google Gemini RAG",
    version="2.0.0",
    lifespan=lifespan
)

# Cấu hình CORS mở rộng cho phép frontend kết nối dễ dàng
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các API Routers
app.include_router(api_config.router)
app.include_router(api_chat.router)
app.include_router(api_search.router)

# Phục vụ Static Files cho Frontend
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        """Truy cập giao diện chính tại đường dẫn gốc /"""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "MedKG-AI Backend đang hoạt động. Thư mục frontend/index.html chưa sẵn sàng."}
        
    @app.get("/{filename}.css")
    async def serve_css(filename: str):
        css_file = FRONTEND_DIR / f"{filename}.css"
        if css_file.exists():
            return FileResponse(str(css_file))
            
    @app.get("/{filename}.js")
    async def serve_js(filename: str):
        js_file = FRONTEND_DIR / f"{filename}.js"
        if js_file.exists():
            return FileResponse(str(js_file))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

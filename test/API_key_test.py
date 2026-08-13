import os
from google import genai
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# Điền API Key của bạn vào đây
API_KEY = os.getenv("GEMINI_API_KEY")  # Hoặc bạn có thể gán trực tiếp: "sk-xxxxxx"

client = genai.Client(api_key=API_KEY)

print("Đang quét danh sách các Model khả dụng cho tài khoản của bạn...")
try:
    for model in client.models.list():
        # Chỉ in ra những model có chữ "gemini"
        if "gemini" in model.name:
            print(f"✅ {model.name}")
            
except Exception as e:
    print(f"Lỗi: {e}")
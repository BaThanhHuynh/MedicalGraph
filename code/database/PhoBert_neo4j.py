"""
[LEGACY — KHÔNG DÙNG CHO BÀI BÁO v10]

Nhúng thế hệ cũ: ghi vào node `Symptom {name}` bằng PhoBERT gốc (C:\\phobert_medical),
lấy vector [CLS], max_length=128.
Bài báo v10 (Mục 5.4) dùng node `TrieuChung {ten_trieu_chung}`, PhoBERT tinh chỉnh
tương phản qua SentenceTransformer, vector 768 chiều đã chuẩn hóa.
Muốn nhúng đúng cấu hình bài báo: dùng code/database/embed_constractive.py.
"""
import json
import torch
import os
from py2neo import Graph, Node
from transformers import AutoTokenizer, AutoModel
from pyvi import ViTokenizer  # Tách từ tiếng Việt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# 1. Cấu hình đường dẫn và kết nối
JSON_FILE = "data/json/ICD10_LLM.json"
MODEL_PATH = "C:\phobert_medical"
# Thay đổi password Neo4j của bạn tại đây
graph = Graph("bolt://localhost:7687", auth=("neo4j", os.getenv("NEO4J_PASSWORD")))  # Thay NEO4J_PASSWORD bằng mật khẩu của bạn

# 2. Load PhoBERT đã fine-tune
print("--- Đang tải PhoBERT model ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

def get_embedding(text):
    # Tách từ để PhoBERT hiểu đúng ngữ nghĩa y khoa (vd: "đau bụng" -> "đau_bụng")
    text_segmented = ViTokenizer.tokenize(text)
    inputs = tokenizer(text_segmented, return_tensors="pt", truncation=True, max_length=128, padding='max_length')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Lấy vector đại diện từ token [CLS]
    embeddings = outputs.last_hidden_state[:, 0, :].flatten().numpy()
    return embeddings.tolist()

# 3. Xử lý file JSON và cập nhật Neo4j
def process_and_import():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tập hợp tất cả triệu chứng duy nhất để tránh tạo lặp node
    unique_symptoms = set()
    for item in data:
        if item.get("symptoms_extract"):
            # Tách các triệu chứng bằng dấu xuống dòng và dọn dẹp khoảng trắng
            slist = [s.strip() for s in item["symptoms_extract"].split("\n") if s.strip()]
            unique_symptoms.update(slist)

    print(f"--- Tìm thấy {len(unique_symptoms)} triệu chứng duy nhất. Bắt đầu nhúng vector ---")

    for i, symptom_name in enumerate(unique_symptoms):
        # Tạo vector
        vector = get_embedding(symptom_name)
        
        # Tạo hoặc cập nhật Node trong Neo4j
        # Dùng MERGE để đảm bảo không tạo trùng nếu chạy lại script
        query = """
        MERGE (s:Symptom {name: $name})
        SET s.embedding = $vector
        RETURN s
        """
        graph.run(query, name=symptom_name, vector=vector)
        
        if (i + 1) % 10 == 0:
            print(f"Đã xử lý: {i + 1}/{len(unique_symptoms)}")

    print("--- Hoàn thành! Toàn bộ triệu chứng đã được nhúng vào MedicalGraph ---")

if __name__ == "__main__":
    process_and_import()
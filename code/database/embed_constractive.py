import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# --- CẤU HÌNH ---
CSV_FILE = "data/medical_data_cleaned.csv"
MODEL_PATH = r"C:\BACKUP_RESEARCH_2026-08-11\models\phobert-medical-constractive"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- 1. TẢI MÔ HÌNH PHOBERT ---
print("--- Đang tải mô hình PhoBERT ---")
model = SentenceTransformer(MODEL_PATH)

def get_embedding(text):
    # Tách từ tiếng Việt chuẩn y khoa
    text_segmented = ViTokenizer.tokenize(text)
    # Nhúng và chuẩn hóa vector (quan trọng cho Contrastive Learning)
    embedding = model.encode(text_segmented, normalize_embeddings=True)
    return embedding.tolist()

# --- 2. XỬ LÝ DỮ LIỆU CSV ---
print("--- Đang xử lý file CSV ---")
try:
    df = pd.read_csv(CSV_FILE, delimiter=';')
    if df.shape[1] == 1: df = pd.read_csv(CSV_FILE, delimiter=',')
except:
    df = pd.read_csv(CSV_FILE, delimiter=',')

# Thu thập tất cả triệu chứng từ cả 2 cột positive và negative
all_symptoms = set()
for col in ['positive_symptoms', 'negative_symptoms']:
    if col in df.columns:
        s_series = df[col].dropna().astype(str)
        for val in s_series:
            # Tách chuỗi triệu chứng bằng dấu |
            parts = [p.strip().lower() for p in val.split('|') if p.strip()]
            all_symptoms.update(parts)

symptoms_list = list(all_symptoms)
print(f"--- Tìm thấy {len(symptoms_list)} triệu chứng duy nhất ---")

# --- 3. CẬP NHẬT VÀO NEO4J ---
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def update_symptom_vectors():
    with driver.session() as session:
        for i, name in enumerate(symptoms_list):
            # Tạo vector 768 chiều
            vector = get_embedding(name)
            
            # Cập nhật hoặc tạo mới Node TrieuChung với thuộc tính embedding
            query = """
            MERGE (t:TrieuChung {ten_trieu_chung: $name})
            SET t.embedding = $vector
            """
            session.run(query, name=name, vector=vector)
            
            if (i + 1) % 50 == 0:
                print(f"Đã nhúng: {i + 1}/{len(symptoms_list)}")

    driver.close()
    print("--- Hoàn tất! Toàn bộ triệu chứng đã được số hóa vector trong Neo4j ---")

if __name__ == "__main__":
    update_symptom_vectors()
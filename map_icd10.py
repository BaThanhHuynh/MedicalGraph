import os
import pandas as pd
import json
import numpy as np
import csv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

ICD_FILE = "data/ICD10_Reference.xlsx" 
INPUT_SYMPTOMS_FILE = "Structured_Extracted_Symptoms.json" 
OUTPUT_FILE = "Final_ICD10_Mapped_By_Name.json"
print("--- BƯỚC 1: KIỂM TRA DỮ LIỆU ĐẦU VÀO ---")

if not os.path.exists(ICD_FILE):
    raise FileNotFoundError(f"Không tìm thấy file danh mục '{ICD_FILE}'. Vui lòng đặt file vào thư mục 'data/'.")

if not os.path.exists(INPUT_SYMPTOMS_FILE):
    raise FileNotFoundError(f"Không tìm thấy file '{INPUT_SYMPTOMS_FILE}'. Bạn cần chạy script trích xuất trước để tạo ra file này.")
print("\nĐã có đủ file dữ liệu. Tiếp tục xử lý...\n")
print("--- BƯỚC 2: TẢI MÔ HÌNH EMBEDDING TIẾNG VIỆT ---")

model = SentenceTransformer(EMBEDDING_MODEL)

def get_embeddings(texts):
    if not texts or len(texts) == 0:
        return np.array([])
    return model.encode(texts, normalize_embeddings=True)

def load_icd_data(filepath):
    def clean_columns_and_check(df):
        df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '')
        if 'Tên bệnh' in df.columns and 'Mã ICD' in df.columns:
            return df
        return None

    try:
        df = pd.read_excel(filepath, dtype=str)
        df_cleaned = clean_columns_and_check(df)
        if df_cleaned is not None:
            return df_cleaned
    except: pass

    delimiters = [',', ';', '\t']
    encodings = ['utf-8-sig', 'utf-8', 'windows-1258', 'latin-1']

    for sep in delimiters:
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=enc, on_bad_lines='skip', dtype=str)
                df_cleaned = clean_columns_and_check(df)
                if df_cleaned is not None:
                    return df_cleaned
            except: continue

    raise Exception("Không thể đọc file ICD-10. Vui lòng kiểm tra lại cấu trúc cột.")

print("\n--- BƯỚC 3: VECTOR HÓA DANH MỤC ICD-10 ---")
try:
    icd_df = load_icd_data(ICD_FILE)
    icd_df = icd_df.dropna(subset=['Tên bệnh', 'Mã ICD'])

    icd_codes = icd_df['Mã ICD'].astype(str).str.strip().tolist()
    icd_names = icd_df['Tên bệnh'].astype(str).str.strip().tolist()

    print(f"Đang nhúng {len(icd_names)} mã ICD-10 thành Vector... (Chờ khoảng 1 phút)")
    icd_embeddings = get_embeddings(icd_names)
    print("Đã hoàn tất Vector hóa ICD-10.")

except Exception as e:
    raise Exception(f"Dừng chương trình do lỗi đọc file: {e}")

def map_icd_by_disease_name(disease_name, top_k=3):
    if not disease_name or str(disease_name).strip() == "":
        return []

    query_vec = get_embeddings([str(disease_name)])
    sim_scores = cosine_similarity(query_vec, icd_embeddings)[0]
    top_indices = np.argsort(sim_scores)[::-1][:top_k]

    candidates = []
    for idx in top_indices:
        candidates.append({
            "icd_code": icd_codes[idx],
            "icd_name": icd_names[idx],
            "confidence_score": round(float(sim_scores[idx]), 4)
        })

    return candidates

print("\n--- BƯỚC 4: BẮT ĐẦU GÁN MÃ ICD-10 BẰNG TÊN BỆNH ---")
try:
    with open(INPUT_SYMPTOMS_FILE, "r", encoding="utf-8") as f:
        patient_data = json.load(f)

    for i, patient in enumerate(tqdm(patient_data)):
        ten_benh = patient.get("ten_benh", "")
        mapped_results = map_icd_by_disease_name(ten_benh, top_k=1)

        if mapped_results:
            patient["icd_code"] = mapped_results[0]["icd_code"]
            patient["icd_name"] = mapped_results[0]["icd_name"]
            patient["confidence"] = mapped_results[0]["confidence_score"]
        else:
            patient["icd_code"] = "None"
            patient["icd_name"] = "Không tìm thấy dữ liệu tên bệnh"
            patient["confidence"] = 0.0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(patient_data, f, ensure_ascii=False, indent=4)

    print(f"\nHOÀN TẤT GÁN MÃ ICD-10!")
    print(f"File kết quả đã được lưu tại: {os.path.abspath(OUTPUT_FILE)}")

except Exception as e:
    print(f"Đã xảy ra lỗi ở Bước 4: {e}")
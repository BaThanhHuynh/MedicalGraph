import pandas as pd
import json
import numpy as np
import os
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

# Đảm bảo các file này nằm CÙNG THƯ MỤC với file code chạy
ICD_FILE = "data/ICD10_Cleaned.csv    "
INPUT_SYMPTOMS_FILE = "json_data/data_with_LLM.json"
OUTPUT_FILE = "ICD10_embedding.json" 

# ==========================================
# 2. KIỂM TRA FILE DỮ LIỆU LOCAL
# ==========================================
print("--- BƯỚC 1: KIỂM TRA DỮ LIỆU ĐẦU VÀO ---")

if not os.path.exists(ICD_FILE):
    print(f"❌ Lỗi: Không tìm thấy file '{ICD_FILE}'. Vui lòng copy file này vào cùng thư mục với code.")
    sys.exit()

if not os.path.exists(INPUT_SYMPTOMS_FILE):
    print(f"❌ Lỗi: Không tìm thấy file '{INPUT_SYMPTOMS_FILE}'. Vui lòng copy file này vào cùng thư mục với code.")
    sys.exit()

print("✅ Đã tìm thấy đủ file dữ liệu. Tiếp tục xử lý...\n")

# ==========================================
# 3. KHỞI TẠO MÔ HÌNH EMBEDDING
# ==========================================
print("--- BƯỚC 2: TẢI MÔ HÌNH EMBEDDING TIẾNG VIỆT ---")
model = SentenceTransformer(EMBEDDING_MODEL)

def get_embeddings(texts):
    if not texts or len(texts) == 0:
        return np.array([])
    return model.encode(texts, normalize_embeddings=True)

# ==========================================
# 4. HÀM ĐỌC FILE THÔNG MINH
# ==========================================
def load_icd_data(filepath):
    def clean_columns_and_check(df):
        df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '')
        if 'Tên bệnh' in df.columns and 'Mã ICD' in df.columns:
            return df
        return None

    # Thử đọc Excel trước
    try:
        df = pd.read_excel(filepath, dtype=str)
        df_cleaned = clean_columns_and_check(df)
        if df_cleaned is not None:
            return df_cleaned
    except: 
        pass

    # Thử đọc CSV với các định dạng khác nhau
    delimiters = [',', ';', '\t']
    encodings = ['utf-8-sig', 'utf-8', 'windows-1258', 'latin-1']

    for sep in delimiters:
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=enc, on_bad_lines='skip', dtype=str)
                df_cleaned = clean_columns_and_check(df)
                if df_cleaned is not None:
                    return df_cleaned
            except: 
                continue

    raise Exception("Không thể đọc file ICD-10. Vui lòng kiểm tra lại cấu trúc cột (Phải có 'Tên bệnh' và 'Mã ICD').")

# ==========================================
# 5. VECTOR HÓA DANH MỤC ICD-10
# ==========================================
print("\n--- BƯỚC 3: VECTOR HÓA DANH MỤC ICD-10 ---")
try:
    icd_df = load_icd_data(ICD_FILE)
    icd_df = icd_df.dropna(subset=['Tên bệnh', 'Mã ICD'])

    icd_codes = icd_df['Mã ICD'].astype(str).str.strip().tolist()
    icd_names = icd_df['Tên bệnh'].astype(str).str.strip().tolist()

    print(f"Đang nhúng {len(icd_names)} mã ICD-10 thành Vector... (Có thể mất 1-2 phút tùy cấu hình máy)")
    icd_embeddings = get_embeddings(icd_names)
    print("✅ Đã hoàn tất Vector hóa ICD-10.")

except Exception as e:
    print(f"❌ Dừng chương trình do lỗi đọc file ICD: {e}")
    sys.exit()

# ==========================================
# 6. THUẬT TOÁN SO KHỚP THEO TÊN BỆNH
# ==========================================
def map_icd_by_disease_name(disease_name, top_k=3):
    # Bỏ qua nếu tên bệnh rỗng
    if not disease_name or str(disease_name).strip() in ["", "None", "nan"]:
        return []

    # Chỉ nhúng TÊN BỆNH
    query_vec = get_embeddings([str(disease_name)])

    # Tính độ tương đồng
    sim_scores = cosine_similarity(query_vec, icd_embeddings)[0]

    # Lấy top cao nhất
    top_indices = np.argsort(sim_scores)[::-1][:top_k]

    candidates = []
    for idx in top_indices:
        candidates.append({
            "icd_code": icd_codes[idx],
            "icd_name": icd_names[idx],
            "confidence_score": round(float(sim_scores[idx]), 4)
        })

    return candidates

# ==========================================
# 7. CHẠY MAPPING CHO TỪNG RECORD
# ==========================================
print("\n--- BƯỚC 4: BẮT ĐẦU GÁN MÃ ICD-10 BẰNG TÊN BỆNH ---")
try:
    with open(INPUT_SYMPTOMS_FILE, "r", encoding="utf-8") as f:
        patient_data = json.load(f)

    for i, patient in enumerate(tqdm(patient_data)):
        # Lấy trực tiếp trường "ten_benh" để đem đi map
        ten_benh = patient.get("ten_benh", "")

        # Gọi hàm mapping
        mapped_results = map_icd_by_disease_name(ten_benh, top_k=1)

        if mapped_results:
            patient["icd_code"] = mapped_results[0]["icd_code"]
            patient["icd_name"] = mapped_results[0]["icd_name"]
            patient["confidence"] = mapped_results[0]["confidence_score"]
        else:
            patient["icd_code"] = "None"
            patient["icd_name"] = "Không tìm thấy dữ liệu tên bệnh"
            patient["confidence"] = 0.0

    # Lưu file trực tiếp vào thư mục hiện tại
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(patient_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ HOÀN TẤT GÁN MÃ ICD-10! Đã lưu kết quả tại file: {OUTPUT_FILE}")

except Exception as e:
    print(f"❌ Đã xảy ra lỗi ở Bước 4: {e}")
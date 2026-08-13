import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from pyvi import ViTokenizer
from sklearn.metrics.pairwise import cosine_similarity

print("Đang tải mô hình PhoBERT gốc (vinai/phobert-base)...")
# Tải Tokenizer và Model gốc từ HuggingFace
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

# Đưa model về chế độ đánh giá (tắt tính toán đạo hàm để tiết kiệm RAM và chạy nhanh hơn)
model.eval()

# Hàm trích xuất đặc trưng vector (Embeddings) bằng PhoBERT
def get_phobert_embedding(text):
    if not text: 
        return np.zeros((1, 768)) # Trả về vector 0 nếu văn bản rỗng (PhoBERT-base có 768 chiều)
    
    # 1. Tiền xử lý: Tách từ tiếng Việt bằng PyVi (Bắt buộc đối với PhoBERT)
    segmented_text = ViTokenizer.tokenize(text)
    
    # 2. Tokenize văn bản để chuyển thành các tensor số học
    encoded = tokenizer(
        segmented_text, 
        padding=True, 
        truncation=True, 
        return_tensors='pt', 
        max_length=256
    )
    
    # 3. Đưa qua mô hình
    with torch.no_grad():
        features = model(**encoded)
    
    # 4. Trích xuất Vector: Lấy vector của token [CLS] (nằm ở vị trí 0) làm đại diện ngữ nghĩa cho toàn bộ câu
    embeddings = features.last_hidden_state[:, 0, :].numpy()
    return embeddings

print("Đang đọc Dữ liệu JSON...")
with open('data/json/ICD10_LLM.json', 'r', encoding='utf-8') as f:
    llm_data = json.load(f)

semantic_scores = []
valid_diseases = 0

print("Đang tính toán Cosine Similarity bằng PhoBERT gốc...")
for item in llm_data:
    disease = str(item.get('ten_benh', '')).strip()
    original_text = str(item.get('trieu_chung_goc', '')).strip()
    symptoms_raw = str(item.get('symptoms_extract', '')).strip()
    
    # Bỏ qua các dòng không hợp lệ
    if not original_text or not symptoms_raw or original_text.lower() == "không có":
        continue
        
    symptoms_list = [s.strip() for s in symptoms_raw.replace('\\n', '\n').split('\n') if s.strip()]
    extracted_text = ", ".join(symptoms_list)
    
    if not extracted_text: continue

    # Chuyển đổi thành Vector qua hàm đã định nghĩa
    vector_original = get_phobert_embedding(original_text)
    vector_extracted = get_phobert_embedding(extracted_text)
    
    # Tính Cosine Similarity
    similarity = cosine_similarity(vector_original, vector_extracted)[0][0]
    
    semantic_scores.append(similarity)
    valid_diseases += 1

# Tính trung bình điểm MINE-2
average_mine2_score = np.mean(semantic_scores)

print("BÁO CÁO MINE-2: ĐÁNH GIÁ CHẤT LƯỢNG NGỮ NGHĨA (PHOBERT GỐC)")
print(f"Tổng số bệnh được đánh giá ngữ nghĩa: {valid_diseases} bệnh")
print(f"► ĐIỂM MINE-2 (Average Cosine Similarity): {average_mine2_score:.4f} / 1.0000")

# Vì PhoBERT gốc chưa fine-tune riêng cho tác vụ so sánh câu, khoảng giá trị vector thường hội tụ cao hơn.
# Chúng ta điều chỉnh nhẹ ngưỡng đánh giá so với các mô hình Sentence-Transformers.
excellent = sum(1 for s in semantic_scores if s >= 0.85)
good = sum(1 for s in semantic_scores if 0.70 <= s < 0.85)
poor = sum(1 for s in semantic_scores if s < 0.70)

print(f"- Mức độ Xuất sắc (>= 0.85) : {excellent} bệnh ({(excellent/valid_diseases):.2%})")
print(f"- Mức độ Tốt (0.70 - 0.84)  : {good} bệnh ({(good/valid_diseases):.2%})")
print(f"- Mức độ Kém (< 0.70)       : {poor} bệnh ({(poor/valid_diseases):.2%})")
print("="*75)
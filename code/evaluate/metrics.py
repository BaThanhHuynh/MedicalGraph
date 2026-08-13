import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyvi import ViTokenizer

print("Đang tải mô hình Sentence-BERT (keepitreal/vietnamese-sbert)...")
model = SentenceTransformer('keepitreal/vietnamese-sbert')

# Hàm tiền xử lý: Tách từ tiếng Việt chuyên dụng
def preprocess_vietnamese(text):
    if not text: return ""
    return ViTokenizer.tokenize(text)

# Hàm phụ trợ trích xuất Triples từ file JSON
def extract_triples_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    triples = []
    nodes = set()
    for item in data:
        disease = str(item.get('ten_benh', '')).strip()
        symptoms_raw = str(item.get('symptoms_extract', '')).strip()
        
        # Bỏ qua nếu dữ liệu rỗng
        if not disease or not symptoms_raw or symptoms_raw.lower() == "không có": 
            continue
            
        nodes.add(disease)
        symptoms = [s.strip() for s in symptoms_raw.replace('\\n', '\n').split('\n') if s.strip()]
        
        for s in symptoms:
            nodes.add(s)
            # Tạo câu mô tả quan hệ và tiền xử lý tách từ ngay lập tức
            sentence = f"{disease} có triệu chứng {s}"
            segmented_sentence = preprocess_vietnamese(sentence)
            triples.append(segmented_sentence)
            
    return list(nodes), triples

print("Đang đọc hai phiên bản Đồ thị (v1.0 và v2.0)...")
# LƯU Ý: Giả sử bạn có 2 file. Thay tên file thành tên thực tế của bạn.
# Ví dụ: ICD10_LLM_v1.json (bản cũ) và ICD10_LLM.json (bản mới cập nhật)
try:
    nodes_v1, triples_v1 = extract_triples_from_json('data/json/ICD10_Medical_Mapping.json')
    nodes_v2, triples_v2 = extract_triples_from_json('data/json/ICD10_LLM.json')
except FileNotFoundError as e:
    print(f"\n[LỖI] Không tìm thấy file: {e}")
    print("Vui lòng đảm bảo bạn có đủ 2 file JSON phiên bản cũ và mới để so sánh!")
    exit()

# ==========================================
# 1. TÍNH TOÁN SYNTACTIC METRICS (CẤU TRÚC)
# ==========================================
delta_nodes = len(nodes_v2) - len(nodes_v1)
delta_edges = len(triples_v2) - len(triples_v1)
growth_rate_edges = (delta_edges / len(triples_v1)) * 100 if len(triples_v1) > 0 else 0

print("\n" + "="*65)
print("1. ĐÁNH GIÁ CẤU TRÚC (SYNTACTIC DYNAMICS)")
print("="*65)
print(f"- Số Nodes (Thực thể) V1 -> V2 : {len(nodes_v1)} -> {len(nodes_v2)} (Thay đổi: {delta_nodes:+} nodes)")
print(f"- Số Edges (Quan hệ) V1 -> V2  : {len(triples_v1)} -> {len(triples_v2)} (Thay đổi: {delta_edges:+} edges)")
print(f"► Tốc độ tăng trưởng đồ thị   : {growth_rate_edges:+.2f}%")

# ==========================================
# 2. TÍNH TOÁN SEMANTIC METRICS (NGỮ NGHĨA)
# ==========================================
print("\nĐang tính toán Vector Ngữ nghĩa tổng thể (Global Semantic Centroid)...")
# Mã hóa toàn bộ triples đã được tách từ thành vector
vectors_v1 = model.encode(triples_v1)
vectors_v2 = model.encode(triples_v2)

# Tính "Trọng tâm" (Centroid) của toàn bộ đồ thị bằng cách lấy trung bình cộng các vector theo trục dọc
centroid_v1 = np.mean(vectors_v1, axis=0).reshape(1, -1)
centroid_v2 = np.mean(vectors_v2, axis=0).reshape(1, -1)

# Tính độ tương đồng giữa 2 trọng tâm đồ thị
semantic_similarity = cosine_similarity(centroid_v1, centroid_v2)[0][0]

print("="*65)
print("2. ĐÁNH GIÁ NGỮ NGHĨA (SEMANTIC DYNAMICS)")
print("="*65)
print(f"► Độ ổn định ngữ nghĩa (Cosine Similarity v1 vs v2): {semantic_similarity:.4f} / 1.0000")
if semantic_similarity > 0.95:
    print("-> Kết luận: Bản cập nhật MỚI giữ vững được chủ đề cốt lõi (Domain Focus) của ngành Y tế.")
elif semantic_similarity > 0.85:
    print("-> Kết luận: Đồ thị có sự chuyển dịch nhẹ về chủ đề nhưng vẫn trong phạm vi chấp nhận được.")
else:
    print("-> Cảnh báo: Đồ thị đang bị chệch hướng nghiêm trọng (Concept Drift). Cần kiểm tra lại nguồn dữ liệu mới nạp vào!")
print("="*65)
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, models
from pyvi import ViTokenizer
import os

# Phải khớp với file setup
CUSTOM_MODEL_PATH = 'C:\phobert_medical'

# Lấy đường dẫn tuyệt đối của thư mục chứa file app.py hiện tại (thư mục Faiss)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_system():
    print(f"[*] Đang tải Custom PhoBERT từ: {CUSTOM_MODEL_PATH}...")
    
    word_embedding_model = models.Transformer(CUSTOM_MODEL_PATH)
    word_embedding_model.max_seq_length = 256 # Cấu hình đồng bộ
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Sử dụng đường dẫn tuyệt đối đã lấy ở trên
    index_path = os.path.join(BASE_DIR, 'custom_phobert_faiss.idx')
    doc_path = os.path.join(BASE_DIR, 'custom_documents.pkl')
    
    print("[*] Đang tải FAISS index (Cosine Similarity)...")
    index = faiss.read_index(index_path)
    
    with open(doc_path, 'rb') as f:
        documents = pickle.load(f)
        
    return model, index, documents

def search(query, model, index, documents, k=5):
    segmented_query = ViTokenizer.tokenize(query)
    query_vector = model.encode([segmented_query]).astype('float32')
    
    # [SỬA ĐỔI 1]: CHUẨN HÓA L2 VECTOR TRUY VẤN
    faiss.normalize_L2(query_vector)
    
    distances, indices = index.search(query_vector, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            doc = documents[idx]
            results.append({
                "rank": i+1,
                "disease": doc['disease'],
                "symptom": doc['symptom'],
                "score": float(distances[0][i]) # [SỬA ĐỔI 2]: Đổi tên biến thành score
            })
    return results

if __name__ == "__main__":
    if not os.path.exists('custom_phobert_faiss.idx'):
        print("⚠️ Vui lòng chạy 'setup_db.py' trước!")
    else:
        model, index, documents = load_system()
        
        print(" HỆ THỐNG TÌM KIẾM VỚI PHOBERT & FAISS")
        
        while True:
            query = input("\nNhập triệu chứng (VD: đau quặn bụng) [Gõ 'exit' để thoát]: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
                
            # Trả về Top 5 bệnh
            results = search(query, model, index, documents, k=5)
            
            print(f"\nKết quả dự đoán (Top 5):")
            for res in results:
                # [SỬA ĐỔI 3]: Cập nhật câu print cho phù hợp với Cosine
                print(f"  [{res['rank']}] Bệnh: {res['disease']} | Độ tương đồng (Cosine): {res['score']:.4f}")
import json
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer, models
from pyvi import ViTokenizer
import os

# --- CẤU HÌNH ĐƯỜNG DẪN MÔ HÌNH CỦA BẠN ---
CUSTOM_MODEL_PATH = 'C:\phobert_medical' 

def create_database():
    print(f"--- ĐANG TẠO DATABASE VỚI CUSTOM PHOBERT TẠI: {CUSTOM_MODEL_PATH} ---")
    
    data_path = 'data/json/ICD10_LLM.json'
    if not os.path.exists(data_path):
        print(f"❌ Lỗi: Không tìm thấy file {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    corpus_embeddings = []

    print("-> Đang load Custom PhoBERT Model...")
    try:
        word_embedding_model = models.Transformer(CUSTOM_MODEL_PATH)
        # BẢO VỆ CHỐNG TRÀN BỘ NHỚ (Cắt ngắn các câu dài hơn 256 token)
        word_embedding_model.max_seq_length = 256 
        
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return

    print(f"-> Đang tiền xử lý, tách từ và mã hóa {len(data)} văn bản...")
    
    for item in data:
        # TỰ ĐỘNG BẮT LỖI KEY KHÔNG TỒN TẠI
        ten_benh = item.get('disease') or item.get('ten_benh') or item.get('Ten_Benh') or ""
        trieu_chung = item.get('symptom') or item.get('symptoms_extract') or item.get('Trieu_Chung') or ""
        
        # Bỏ qua dòng trống
        if not ten_benh or not trieu_chung or str(trieu_chung).lower() == "không có":
            continue

        combined_text = f"{ten_benh} : {trieu_chung}"
        documents.append({
            'disease': ten_benh,
            'symptom': trieu_chung
        })
        
        # Tách từ tiếng Việt
        segmented_text = ViTokenizer.tokenize(combined_text)
        vec = model.encode(segmented_text)
        corpus_embeddings.append(vec)

    # ÉP KIỂU SANG NUMPY ARRAY ĐỂ FAISS ĐỌC ĐƯỢC
    corpus_embeddings = np.array(corpus_embeddings).astype('float32')

    # [SỬA ĐỔI 1]: CHUẨN HÓA L2 CHO COSINE SIMILARITY
    faiss.normalize_L2(corpus_embeddings)

    # [SỬA ĐỔI 2]: DÙNG INDEX INNER PRODUCT THAY VÌ L2
    embedding_dim = corpus_embeddings.shape[1] 
    index = faiss.IndexFlatIP(embedding_dim) # Thay L2 bằng IP
    index.add(corpus_embeddings)

    print(f"-> Kích thước vector: {embedding_dim} chiều")
    print(f"-> Đã index {index.ntotal} mẫu.")

    # Lưu FAISS Index và dữ liệu gốc
    faiss.write_index(index, 'custom_phobert_faiss.idx')
    with open('custom_documents.pkl', 'wb') as f:
        pickle.dump(documents, f)
    
    print("✅ >>> HOÀN TẤT SETUP DATABASE VỚI COSINE SIMILARITY <<<")

if __name__ == "__main__":
    create_database()
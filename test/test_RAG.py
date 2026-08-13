import os
import torch
import torch.nn.functional as F
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
from pyvi import ViTokenizer
import logging
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # Thay bằng pass của em
PHOBERT_PATH = r"C:\phobert_medical" 

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 2. KHỞI TẠO
# ==========================================
print("[*] Đang kết nối tới đồ thị Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

print(f"[*] Đang tải Custom PhoBERT từ: {PHOBERT_PATH}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_PATH, local_files_only=True)
model = AutoModel.from_pretrained(PHOBERT_PATH, local_files_only=True).to(device)
model.eval()

# ==========================================
# 3. HÀM XỬ LÝ LÕI (HYBRID GRAPHRAG)
# ==========================================
def get_batch_embeddings(symptom_list):
    """Nhúng TỪNG triệu chứng thành TỪNG vector riêng biệt"""
    if not symptom_list: return []
    text_segmented = [ViTokenizer.tokenize(text) for text in symptom_list]
    inputs = tokenizer(
        text_segmented, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    embeddings = outputs.last_hidden_state[:, 0, :]
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    return normalized_embeddings.tolist()

def retrieve_diseases_via_graph(user_symptoms):
    vectors = get_batch_embeddings(user_symptoms)
    num_input_symptoms = len(user_symptoms)
    
    # CYPHER ĐÃ BỔ SUNG ĐIỂM COSINE (HYBRID LOGIC)
    cypher_query = """
    // BƯỚC 1: Dùng Vector để tìm Node Triệu chứng
    UNWIND $vectors AS vec
    CALL db.index.vector.queryNodes('symptom_vector_index', 3, vec) 
    YIELD node AS matched_symptom, score
    WHERE score > 0.85 
    
    // BƯỚC 2: Giữ lại điểm Cosine cao nhất cho mỗi triệu chứng (tránh nhiễu nếu nhập lặp từ)
    WITH matched_symptom, max(score) AS sym_cosine_score
    
    // BƯỚC 3: Dùng Đồ thị truy ngược về Bệnh
    MATCH (disease:Disease)-[:HAS_SYMPTOM]->(matched_symptom)
    
    // BƯỚC 4: Tính toán hỗn hợp cả Đồ thị (Count) và Vector (Cosine)
    WITH disease, 
         collect(matched_symptom.name) AS matched_sym_names, 
         count(matched_symptom) AS match_count,
         avg(sym_cosine_score) AS avg_cosine_score // Tính điểm Cosine trung bình của các triệu chứng khớp
         
    MATCH (disease)-[:HAS_SYMPTOM]->(all_d_syms:Symptom)
    WITH disease, matched_sym_names, match_count, avg_cosine_score, count(all_d_syms) AS total_disease_syms
    
    RETURN disease.name AS disease_name, 
           matched_sym_names,
           match_count,
           avg_cosine_score,
           (tofloat(match_count) / total_disease_syms) * 100 AS disease_specificity
           
    // XẾP HẠNG: (1) Số triệu chứng khớp -> (2) Điểm Cosine trung bình -> (3) Độ đặc hiệu của bệnh
    ORDER BY match_count DESC, avg_cosine_score DESC, disease_specificity DESC
    LIMIT 5
    """
    
    results = []
    with driver.session() as session:
        records = session.run(cypher_query, vectors=vectors, num_inputs=num_input_symptoms)
        for record in records:
            results.append({
                "disease": record["disease_name"],
                "matched_symptoms": record["matched_sym_names"],
                "match_count": record["match_count"],
                "cosine_score": record["avg_cosine_score"],
                "specificity": record["disease_specificity"]
            })
    return results

# ==========================================
# 4. GIAO DIỆN CHẠY CLI
# ==========================================
if __name__ == "__main__":
    print(" HỆ THỐNG TÌM KIẾM VỚI PHOBERT & GRAPHRAG")
    
    try:
        while True:
            test_input = input("\nNhập triệu chứng (Cách nhau bởi dấu phẩy) [Gõ 'exit' để thoát]: ")
            
            if test_input.lower() in ['quit', 'exit', 'thoát']:
                break
                
            if not test_input.strip():
                continue

            user_symptoms = [s.strip() for s in test_input.split(',') if s.strip()]
            
            if not user_symptoms: 
                continue

            results = retrieve_diseases_via_graph(user_symptoms)
            
            print(f"\nKết quả dự đoán (Top 5):")
            if not results:
                print("  Không tìm thấy bệnh nào phù hợp. (Vui lòng thử mô tả triệu chứng khác)")
            else:
                for i, res in enumerate(results, 1):
                    symptoms_str = ", ".join(res['matched_symptoms'])
                    print(f"  [{i}] Bệnh: {res['disease']}")
                    # Hiển thị điểm Cosine rõ ràng cho em đối chiếu
                    print(f"Độ tương đồng ngữ nghĩa (Cosine): {res['cosine_score']:.4f}")
                    print()
                    
    except KeyboardInterrupt:
        pass
    finally:
        driver.close()

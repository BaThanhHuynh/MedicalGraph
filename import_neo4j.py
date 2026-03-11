import json
from neo4j import GraphDatabase
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH KẾT NỐI LOCALHOST
# ==========================================
NEO4J_URI = "neo4j://localhost:7687" 
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "21102005" # THAY BẰNG PASSWORD BẠN VỪA ĐẶT TRÊN NEO4J DESKTOP

INPUT_JSON_FILE = "Final_ICD10_Mapped_By_Name.json" 

# ==========================================
# 2. KHỞI TẠO DRIVER & KIỂM TRA
# ==========================================
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

try:
    driver.verify_connectivity()
    print("✅ Đã kết nối thành công với Neo4j Desktop!")
except Exception as e:
    raise Exception(f"❌ Lỗi kết nối (Hãy chắc chắn bạn đã bấm nút Start trên Neo4j Desktop): {e}")

# ==========================================
# 3. HÀM TẠO GRAPH
# ==========================================
def create_graph_data(tx, record):
    ten_benh = record.get("ten_benh", "").strip()
    if not ten_benh: return

    # 1. Thực thể Bệnh
    tx.run("MERGE (d:Disease {name: $name})", name=ten_benh)

    # 2. Thực thể ICD-10
    icd_code = record.get("icd_code", "None")
    icd_name = record.get("icd_name", "")
    confidence = record.get("confidence", 0.0)

    if icd_code != "None":
        tx.run("""
            MATCH (d:Disease {name: $disease_name})
            MERGE (i:ICD10 {code: $icd_code})
            ON CREATE SET i.name = $icd_name
            MERGE (d)-[r:MAPPED_TO]->(i)
            SET r.confidence = $confidence
        """, disease_name=ten_benh, icd_code=icd_code, icd_name=icd_name, confidence=confidence)

    # 3. Thực thể Triệu chứng
    symptoms_str = record.get("symptoms_extract", "")
    symptoms_list = [s.strip() for s in symptoms_str.split("\n") if s.strip()]
    
    for sym in symptoms_list:
        tx.run("""
            MATCH (d:Disease {name: $disease_name})
            MERGE (s:Symptom {name: $symptom_name})
            MERGE (d)-[:HAS_SYMPTOM]->(s)
        """, disease_name=ten_benh, symptom_name=sym)

    # 4. Thực thể Phủ định
    negated_str = record.get("negated_symptoms", "")
    negated_list = [s.strip() for s in negated_str.split("\n") if s.strip()]
    
    for n_sym in negated_list:
        tx.run("""
            MATCH (d:Disease {name: $disease_name})
            MERGE (s:Symptom {name: $symptom_name})
            MERGE (d)-[:HAS_NEG_SYMPTOM]->(s)
        """, disease_name=ten_benh, symptom_name=n_sym)

# ==========================================
# 4. TIẾN HÀNH IMPORT
# ==========================================
with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Đang đổ dữ liệu vào Đồ thị...")
with driver.session() as session:
    # Xóa sạch data cũ nếu có
    session.run("MATCH (n) DETACH DELETE n") 
    
    for record in tqdm(data):
        session.execute_write(create_graph_data, record)

print("\n✅ HOÀN TẤT! HÃY MỞ NEO4J BROWSER ĐỂ XEM KẾT QUẢ.")
driver.close()
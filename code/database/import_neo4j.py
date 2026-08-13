"""
[LEGACY — KHÔNG DÙNG CHO BÀI BÁO v10]

Script này dựng schema thế hệ cũ (nhãn tiếng Anh):
    Disease / Symptom / ICD10 + HAS_SYMPTOM, HAS_NEG_SYMPTOM, MAPPED_TO
Bài báo v10 (Mục 3.1) dùng schema 5 lớp / 5 quan hệ tiếng Việt:
    BenhLy / TrieuChung / ICD10 / ChuyenKhoa / XetNghiem
    + CO_TRIEU_CHUNG, LOAI_TRU_TRIEU_CHUNG, CO_MA_ICD10,
      THUOC_CHUYEN_KHOA, DUOC_CHAN_DOAN_BANG

CẢNH BÁO: script bắt đầu bằng `MATCH (n) DETACH DELETE n` — chạy nhầm sẽ xóa sạch
đồ thị v2 rồi dựng lại schema cũ, khiến medkgbert_dx.py không truy vấn được.
Muốn dựng đồ thị của bài báo: dùng code/database/import_neo4j_v2.py.
"""
import json
import os
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# ==========================================
# 1. CẤU HÌNH KẾT NỐI LOCALHOST
# ==========================================
NEO4J_URI = "neo4j://localhost:7687" 
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") # THAY BẰNG PASSWORD BẠN VỪA ĐẶT TRÊN NEO4J DESKTOP

INPUT_JSON_FILE = "data/json/ICD10_LLM.json" 

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

    # 3. Thực thể Triệu chứng (CÓ TRỌNG SỐ)
    symptoms_str = record.get("symptoms_extract", "")
    symptoms_list = [s.strip() for s in symptoms_str.split("\n") if s.strip()]
    
    default_weight = 1.0 # [SỬA ĐỔI 1]: Trọng số mặc định
    
    for sym in symptoms_list:
        tx.run("""
            MATCH (d:Disease {name: $disease_name})
            MERGE (s:Symptom {name: $symptom_name})
            MERGE (d)-[r:HAS_SYMPTOM]->(s)
            ON CREATE SET r.weight = $weight
            ON MATCH SET r.weight = $weight
        """, disease_name=ten_benh, symptom_name=sym, weight=default_weight)

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
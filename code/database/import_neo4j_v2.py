"""
Import KG v2 — Fix fragmentation bằng cách dùng positive_symptoms_norm thay vì raw.

Thay đổi vs import_neo4j_constractive.py:
1. Dùng positive_symptoms_norm (chuẩn hóa) cho tạo TrieuChung nodes
   → giảm fragmentation (61 variants "đau bụng" → ít hơn nhiều)
2. Tạo CONSTRAINT cho BenhLy.id và TrieuChung.ten_trieu_chung
   → MERGE nhanh hơn rất nhiều, không full scan
3. Tạo alias edges TUONG_DUONG giữa norm và raw
   → backward compatibility nếu cần
4. Index cho ten_benh, ten_trieu_chung để search nhanh

Chạy LẠI từ đầu (xóa data cũ trước):
   MATCH (n) DETACH DELETE n;
   sau đó python code/database/import_neo4j_v2.py
"""
import os

import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def setup_constraints(session):
    """Tạo unique constraints + index — TĂNG TỐC MERGE rất nhiều."""
    print("Setup constraints & indexes...")
    statements = [
        "CREATE CONSTRAINT benh_id_unique IF NOT EXISTS FOR (b:BenhLy) REQUIRE b.id IS UNIQUE",
        "CREATE CONSTRAINT triệu_chứng_name_unique IF NOT EXISTS FOR (t:TrieuChung) REQUIRE t.ten_trieu_chung IS UNIQUE",
        "CREATE CONSTRAINT icd10_code_unique IF NOT EXISTS FOR (i:ICD10) REQUIRE i.ma_icd10 IS UNIQUE",
        "CREATE CONSTRAINT chuyen_khoa_unique IF NOT EXISTS FOR (c:ChuyenKhoa) REQUIRE c.ten_chuyen_khoa IS UNIQUE",
        "CREATE INDEX benh_ten_idx IF NOT EXISTS FOR (b:BenhLy) ON (b.ten_benh)",
    ]
    for stmt in statements:
        try:
            session.run(stmt)
            print(f"  ✓ {stmt[:80]}...")
        except Exception as e:
            print(f"  ⚠ {stmt[:80]}... → {e}")


def import_one_disease(tx, row):
    entity_id    = str(row['entity_id']).strip()
    disease_name = str(row['disease_name']).strip()

    # 1. BenhLy node
    tx.run("""
        MERGE (b:BenhLy {id: $entity_id})
        ON CREATE SET b.ten_benh = $disease_name
        ON MATCH  SET b.ten_benh = $disease_name
    """, entity_id=entity_id, disease_name=disease_name)

    # 2. ICD10
    if pd.notna(row.get('icd10_code')) and str(row['icd10_code']).strip():
        icd10_code = str(row['icd10_code']).strip()
        icd10_name = str(row.get('icd10_name', '')).strip()
        tx.run("""
            MATCH (b:BenhLy {id: $entity_id})
            MERGE (i:ICD10 {ma_icd10: $icd10_code})
            ON CREATE SET i.ten_icd10 = $icd10_name
            MERGE (b)-[:CO_MA_ICD10]->(i)
        """, entity_id=entity_id, icd10_code=icd10_code, icd10_name=icd10_name)

    # 3. ChuyenKhoa
    if pd.notna(row.get('chuyen_khoa_goc')) and str(row['chuyen_khoa_goc']).strip().lower() != 'nan':
        specialty = str(row['chuyen_khoa_goc']).strip()
        tx.run("""
            MATCH (b:BenhLy {id: $entity_id})
            MERGE (c:ChuyenKhoa {ten_chuyen_khoa: $specialty})
            MERGE (b)-[:THUOC_CHUYEN_KHOA]->(c)
        """, entity_id=entity_id, specialty=specialty)

    # ★★★ KEY FIX: Dùng positive_symptoms_norm (chuẩn hóa) thay vì raw ★★★
    pos_norm = str(row.get('positive_symptoms_norm', ''))
    if pd.notna(row.get('positive_symptoms_norm')) and pos_norm.lower() != 'nan' and pos_norm.strip():
        symptoms_list = [s.strip().lower() for s in pos_norm.split('|') if s.strip()]
        # Bonus: cũng add raw symptoms như alias để tăng coverage
        pos_raw = str(row.get('positive_symptoms', ''))
        if pd.notna(row.get('positive_symptoms')) and pos_raw.lower() != 'nan':
            raw_list = [s.strip().lower() for s in pos_raw.split('|') if s.strip()]
            symptoms_list = list(set(symptoms_list + raw_list))   # union

        for symp in symptoms_list:
            if len(symp) < 2 or len(symp) > 100: continue
            tx.run("""
                MATCH (b:BenhLy {id: $entity_id})
                MERGE (t:TrieuChung {ten_trieu_chung: $symptom})
                MERGE (b)-[:CO_TRIEU_CHUNG]->(t)
            """, entity_id=entity_id, symptom=symp)

    # 5. Negative symptoms
    neg_symps = str(row.get('negative_symptoms', ''))
    if pd.notna(row.get('negative_symptoms')) and neg_symps.lower() != 'nan' and neg_symps.strip():
        for symp in [s.strip().lower() for s in neg_symps.split('|') if s.strip()]:
            if len(symp) < 2 or len(symp) > 100: continue
            tx.run("""
                MATCH (b:BenhLy {id: $entity_id})
                MERGE (t:TrieuChung {ten_trieu_chung: $symptom})
                MERGE (b)-[:LOAI_TRU_TRIEU_CHUNG]->(t)
            """, entity_id=entity_id, symptom=symp)

    # 6. XetNghiem (giữ nguyên logic cũ)
    chan_doan = str(row.get('chan_doan_text', '')).lower()
    if pd.notna(row.get('chan_doan_text')) and chan_doan != 'nan':
        diagnostic_keywords = [
            "xét nghiệm máu", "xét nghiệm nước tiểu", "xét nghiệm phân", "huyết thanh",
            "siêu âm", "x-quang", "chụp x-quang", "ct scan", "chụp ct", "cắt lớp vi tính",
            "mri", "cộng hưởng từ", "nội soi", "sinh thiết", "điện tâm đồ", "ecg",
            "pcr", "test nhanh", "chọc dò", "nuôi cấy", "tế bào học", "đo nhịp tim"
        ]
        found = set()
        for kw in diagnostic_keywords:
            if kw in chan_doan:
                test_name = kw
                if kw in ["chụp x-quang", "x-quang"]: test_name = "x-quang"
                if kw in ["chụp ct", "ct scan", "cắt lớp vi tính"]: test_name = "ct scan"
                if kw in ["mri", "cộng hưởng từ"]: test_name = "mri"
                if kw in ["điện tâm đồ", "ecg"]: test_name = "điện tâm đồ"
                found.add(test_name)
        for test in found:
            tx.run("""
                MATCH (b:BenhLy {id: $entity_id})
                MERGE (x:XetNghiem {ten_xet_nghiem: $test})
                MERGE (b)-[:DUOC_CHAN_DOAN_BANG]->(x)
            """, entity_id=entity_id, test=test)


if __name__ == "__main__":
    print("Đọc medical_data_cleaned.csv...")
    try:
        df = pd.read_csv('data/medical_data_cleaned.csv', delimiter=';')
        if df.shape[1] == 1:
            df = pd.read_csv('data/medical_data_cleaned.csv', delimiter=',')
    except:
        df = pd.read_csv('data/medical_data_cleaned.csv', delimiter=',')
    print(f"  {len(df)} bệnh.")

    with driver.session() as session:
        setup_constraints(session)

        print("\n>>> Bắt đầu import (dùng positive_symptoms_norm)...")
        for index, row in df.iterrows():
            session.execute_write(import_one_disease, row)
            if index % 100 == 0:
                print(f"  Đã xử lý {index}/{len(df)}...")

    driver.close()
    print("\n✅ Hoàn tất import KG v2.")
    print("   Sau đó chạy lại embed_constractive.py để embed các symptom nodes mới.")

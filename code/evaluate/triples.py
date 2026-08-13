import pandas as pd
import json
import csv

print("Đang đọc Dữ liệu Gốc (Full KG) từ file JSON và file Manual...")

# ==========================================
# 1. ĐỌC DỮ LIỆU TỪ FILE JSON (FULL KNOWLEDGE GRAPH)
# ==========================================
with open('data/json/ICD10_LLM.json', 'r', encoding='utf-8') as f:
    llm_data = json.load(f)

kg_triples_raw = set()
for item in llm_data:
    disease = str(item.get('ten_benh', '')).strip().lower()
    if not disease: continue
        
    symptoms_raw = str(item.get('symptoms_extract', ''))
    
    # Xử lý dấu xuống dòng trong JSON (\n hoặc \\n)
    if '\\n' in symptoms_raw:
        symptoms = [s.strip().lower() for s in symptoms_raw.split('\\n') if s.strip()]
    elif '\n' in symptoms_raw:
        symptoms = [s.strip().lower() for s in symptoms_raw.split('\n') if s.strip()]
    else:
        symptoms = [symptoms_raw.strip().lower()] if symptoms_raw.strip() else []
        
    for symp in symptoms:
        # TẠO TRIPLE CHUẨN: (Subject, Predicate, Object)
        kg_triples_raw.add((disease, 'HAS_SYMPTOM', symp))

# ==========================================
# 2. ĐỌC FILE MANUAL (TIÊU CHUẨN VÀNG)
# ==========================================
def read_robust_csv(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if not row: continue
            if len(row) > 4:
                symptom_text = ";".join(row[3:])
                row = row[:3] + [symptom_text]
            elif len(row) < 4:
                row += [''] * (4 - len(row))
            data.append(row)
    
    if len(data) > 0:
        header = ["Ma_ICD10", "Ten_Benh", "Quan_He", "Trieu_Chung"]
        start_idx = 1 if data[0][1].lower() == 'ten_benh' else 0
        return pd.DataFrame(data[start_idx:], columns=header)
    return pd.DataFrame()

df_manual = read_robust_csv('gold_standard_manual.csv')

manual_triples_raw = set()
for _, row in df_manual.iterrows():
    disease = str(row.get('Ten_Benh', '')).strip().lower()
    if not disease or disease == 'nan': continue
    
    # Lấy Mối quan hệ (Predicate) từ file, mặc định là HAS_SYMPTOM nếu trống
    relation = str(row.get('Quan_He', 'HAS_SYMPTOM')).strip().upper()
    if not relation or relation == 'NAN': relation = 'HAS_SYMPTOM'
    
    symptoms_raw = str(row.get('Trieu_Chung', ''))
    symptoms_lines = symptoms_raw.split('\n')
    for s in symptoms_lines:
        for sub_s in s.split(';'):
            symp_clean = sub_s.strip().lower()
            if symp_clean:
                # TẠO TRIPLE CHUẨN: (Subject, Predicate, Object)
                manual_triples_raw.add((disease, relation, symp_clean))

# ==========================================
# 3. LỌC ĐỂ SO SÁNH CÔNG BẰNG (TRÊN TẬP BỆNH TEST)
# ==========================================
# Trích xuất danh sách bệnh từ vị trí Subject (vị trí 0) của bộ ba
manual_diseases = set([subject for subject, predicate, obj in manual_triples_raw])

# Lọc file JSON: Chỉ giữ lại các Triples thuộc các bệnh có trong file test
kg_triples = set([(subj, pred, obj) for subj, pred, obj in kg_triples_raw if subj in manual_diseases])
manual_triples = manual_triples_raw

print(f"\nĐang đánh giá {len(manual_diseases)} bệnh bằng thuật toán Substring Matching (Chuẩn Triples)...")

# ==========================================
# 4. ĐỐI SÁNH BỘ BA (TRIPLE MATCHING)
# ==========================================
correct_kg_triples = set()
correct_manual_triples = set()

# Bóc tách 3 biến: Chủ thể (d), Mối quan hệ (rel), Khách thể (s)
for d_kg, rel_kg, s_kg in kg_triples:
    for d_man, rel_man, s_man in manual_triples:
        
        # ĐIỀU KIỆN 1: Bệnh phải giống nhau VÀ Mối quan hệ phải khớp nhau
        if d_kg == d_man and rel_kg == rel_man:
            
            # ĐIỀU KIỆN 2: Khách thể (Triệu chứng) khớp một phần (Substring)
            if s_kg in s_man or s_man in s_kg:
                correct_kg_triples.add((d_kg, rel_kg, s_kg))
                correct_manual_triples.add((d_man, rel_man, s_man))

# Các phần sai / dư thừa / thiếu
incorrect_kg_triples = kg_triples - correct_kg_triples
missing_manual_triples = manual_triples - correct_manual_triples

accuracy = len(correct_kg_triples) / len(kg_triples) if len(kg_triples) > 0 else 0
completeness = len(correct_manual_triples) / len(manual_triples) if len(manual_triples) > 0 else 0

# ==========================================
# 5. IN BÁO CÁO KẾT QUẢ CHÍNH THỨC
# ==========================================
print(f"1. Tổng số bệnh được đối chiếu:              {len(manual_diseases)} bệnh")
print(f"2. Tổng khối lượng Triples của KG gốc:       {len(kg_triples)}")
print(f"3. Tổng khối lượng Triples chuẩn (Manual):   {len(manual_triples)}")
print(f"4. Số lượng Triples KG bắt trúng thực tế:    {len(correct_manual_triples)}")
print(f"ĐỘ CHÍNH XÁC (Accuracy/Precision): {accuracy:.2%} (Mức độ đáng tin cậy của KG)")
print(f"ĐỘ ĐẦY ĐỦ (Completeness/Recall)  : {completeness:.2%} (Khả năng bao quát kiến thức)")

# ==========================================
# 6. XUẤT FILE PHÂN TÍCH LỖI
# ==========================================
report_data = []

# Xuất đủ 3 thành phần của Triple ra file
for d, r, s in correct_kg_triples:
    report_data.append({'Ten_Benh': d, 'Quan_He': r, 'Trieu_Chung': s, 'Phan_Loai': 'ĐÚNG'})
for d, r, s in incorrect_kg_triples:
    report_data.append({'Ten_Benh': d, 'Quan_He': r, 'Trieu_Chung': s, 'Phan_Loai': 'DƯ / NHIỄU'})
for d, r, s in missing_manual_triples:
    report_data.append({'Ten_Benh': d, 'Quan_He': r, 'Trieu_Chung': s, 'Phan_Loai': 'BỎ SÓT'})

df_report = pd.DataFrame(report_data).sort_values(by=['Ten_Benh', 'Phan_Loai'])
df_report.to_csv('Official_KG_Triple_Evaluation.csv', index=False, sep=';', encoding='utf-8-sig')

print("\nĐã xuất file Error Analysis bao gồm cột 'Quan_He' (Relation): 'Official_KG_Triple_Evaluation.csv'")
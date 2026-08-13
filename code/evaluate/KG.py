import pandas as pd
import json
import re

print("Đang đọc Dữ liệu Đồ thị Tri thức (Full KG) từ file JSON...")

# Đọc dữ liệu
with open('data/json/ICD10_LLM.json', 'r', encoding='utf-8') as f:
    llm_data = json.load(f)

consistency_errors = []
redundancy_cases = []
total_triples = 0
total_diseases = len(llm_data)

print(f"Bắt đầu quét {total_diseases} bệnh để tìm lỗi logic và dư thừa...\n")

for item in llm_data:
    disease = str(item.get('ten_benh', '')).strip().lower()
    icd_code = str(item.get('icd_code', '')).strip().upper()
    symptoms_raw = str(item.get('symptoms_extract', ''))

    # ==========================================
    # 1. KIỂM TRA TÍNH NHẤT QUÁN (CONSISTENCY)
    # ==========================================
    
    # Rule 1: Kiểm tra định dạng mã ICD-10 (Bắt đầu bằng A-Z, theo sau là số)
    if icd_code and not re.match(r'^[A-Z][0-9]', icd_code):
        consistency_errors.append({
            'Ten_Benh': disease, 
            'Loi_Nhat_Quan': 'Sai định dạng chuẩn ICD-10', 
            'Chi_Tiet': icd_code
        })

    # Xử lý list triệu chứng
    if '\\n' in symptoms_raw:
        symptoms = [s.strip().lower() for s in symptoms_raw.split('\\n') if s.strip()]
    elif '\n' in symptoms_raw:
        symptoms = [s.strip().lower() for s in symptoms_raw.split('\n') if s.strip()]
    else:
        symptoms = [symptoms_raw.strip().lower()] if symptoms_raw.strip() else []

    # Loại bỏ trùng lặp tuyệt đối (Exact match duplicate) - Bản thân hàm set() đã giải quyết một phần redundancy
    unique_symptoms = list(set(symptoms))
    total_triples += len(unique_symptoms)

    for symp in unique_symptoms:
        # Rule 2: Thực thể rỗng hoặc chứa ký tự rác
        if len(symp) < 2 or symp in ['có', 'không', 'bị', 'là']:
            consistency_errors.append({
                'Ten_Benh': disease, 
                'Loi_Nhat_Quan': 'Thực thể quá ngắn hoặc là từ nhiễu (Stopword)', 
                'Chi_Tiet': symp
            })
            
        # Rule 3: Vi phạm logic Ontology (Self-loop: Bệnh tự là triệu chứng của chính mình)
        elif disease == symp:
            consistency_errors.append({
                'Ten_Benh': disease, 
                'Loi_Nhat_Quan': 'Vòng lặp tự chỉ (Self-loop)', 
                'Chi_Tiet': f"({disease}) -[HAS_SYMPTOM]-> ({symp})"
            })

    # ==========================================
    # 2. KIỂM TRA ĐỘ DƯ THỪA (REDUNDANCY)
    # ==========================================
    
    # Rule 4: Phát hiện triệu chứng bao hàm nhau trong CÙNG MỘT bệnh
    # Ví dụ: Bệnh A có cả "sốt" và "sốt cao" -> Dư thừa node "sốt"
    for i in range(len(unique_symptoms)):
        for j in range(i + 1, len(unique_symptoms)):
            s1 = unique_symptoms[i]
            s2 = unique_symptoms[j]
            
            # Chỉ xét các cụm từ có nghĩa (độ dài > 3)
            if len(s1) > 3 and len(s2) > 3:
                # Nếu s1 nằm trong s2 hoặc ngược lại
                if s1 in s2 or s2 in s1:
                    redundancy_cases.append({
                        'Ten_Benh': disease,
                        'Trieu_Chung_1': s1,
                        'Trieu_Chung_2': s2,
                        'Kieu_Du_Thua': 'Bao hàm ngữ nghĩa (Substring Overlap)'
                    })

# ==========================================
# 3. TÍNH TOÁN CHỈ SỐ VÀ IN BÁO CÁO
# ==========================================
# Tính tỷ lệ lỗi
consistency_error_rate = len(consistency_errors) / total_triples if total_triples > 0 else 0
consistency_score = 1 - consistency_error_rate # Điểm Nhất quán (Càng gần 100% càng tốt)

redundancy_rate = len(redundancy_cases) / total_triples if total_triples > 0 else 0

print(f"Tổng số Triples trong hệ thống:       {total_triples}")
print(f"Tính nhất quán (Consistency):      {consistency_score:.2%} (Mức độ sạch sẽ, không mâu thuẫn)")
print(f"   - Số lỗi logic phát hiện:          {len(consistency_errors)} lỗi")
print(f"Độ dư thừa(Redundancy):           {redundancy_rate:.2%} (Tỷ lệ thông tin bị lặp/bao hàm)")
print(f"   - Số cặp triệu chứng bị dư thừa:   {len(redundancy_cases)} cặp")

# Xuất file chi tiết để xem xét
if consistency_errors:
    df_consistency = pd.DataFrame(consistency_errors)
    df_consistency.to_csv('KG_Consistency_Errors.csv', index=False, sep=';', encoding='utf-8-sig')
    print("Đã xuất danh sách lỗi Nhất quán ra file: 'KG_Consistency_Errors.csv'")

if redundancy_cases:
    df_redundancy = pd.DataFrame(redundancy_cases)
    df_redundancy.to_csv('KG_Redundancy_Cases.csv', index=False, sep=';', encoding='utf-8-sig')
    print("Đã xuất danh sách các node Dư thừa ra file: 'KG_Redundancy_Cases.csv'")
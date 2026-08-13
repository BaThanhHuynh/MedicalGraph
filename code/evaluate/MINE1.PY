import pandas as pd
import json
import csv

print("Đang đánh giá MINE-1 (Knowledge Retention)...")

# 1. Đọc dữ liệu JSON
with open('data/json/ICD10_LLM.json', 'r', encoding='utf-8') as f:
    llm_data = json.load(f)

kg_triples = set()
for item in llm_data:
    disease = str(item.get('ten_benh', '')).strip().lower()
    if not disease: continue
    symptoms_raw = str(item.get('symptoms_extract', ''))
    symptoms = [s.strip().lower() for s in symptoms_raw.replace('\\n', '\n').split('\n') if s.strip()]
    for symp in symptoms:
        kg_triples.add((disease, symp))

# 2. Đọc file Manual
def read_manual(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if not row: continue
            if len(row) > 4: row = row[:3] + [";".join(row[3:])]
            elif len(row) < 4: row += [''] * (4 - len(row))
            data.append(row)
    return pd.DataFrame(data[1:], columns=["Ma_ICD10", "Ten_Benh", "Quan_He", "Trieu_Chung"]) if len(data) > 1 else pd.DataFrame()

df_manual = read_manual('gold_standard_manual.csv')
manual_triples = set()
for _, row in df_manual.iterrows():
    disease = str(row.get('Ten_Benh', '')).strip().lower()
    if not disease or disease == 'nan': continue
    symptoms = [s.strip().lower() for s in str(row.get('Trieu_Chung', '')).split('\n') if s.strip()]
    for s in symptoms:
        for sub_s in s.split(';'):
            if sub_s.strip(): manual_triples.add((disease, sub_s.strip()))

# Lọc các bệnh chung để đánh giá
manual_diseases = set([d for d, s in manual_triples])
kg_triples = set([(d, s) for d, s in kg_triples if d in manual_diseases])

# 3. Tính toán MINE-1
correct = set()
for d_kg, s_kg in kg_triples:
    for d_man, s_man in manual_triples:
        if d_kg == d_man and (s_kg in s_man or s_man in s_kg):
            correct.add((d_kg, s_kg))
            break # Tính là 1 lần trúng

precision = len(correct) / len(kg_triples) if kg_triples else 0
recall = len(correct) / len(manual_triples) if manual_triples else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision  : {precision:.2%}")
print(f"Recall        : {recall:.2%}")
print(f"F1-Score    : {f1_score:.2%} (Chỉ số tổng hợp mức độ giữ lại thông tin)")

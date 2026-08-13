import json
import random
import pandas as pd

print("Đang đọc Dữ liệu Đồ thị thực tế...")
with open('json_data/ICD10_LLM.json', 'r', encoding='utf-8') as f:
    llm_data = json.load(f)

all_triples = []
for item in llm_data:
    disease = str(item.get('ten_benh', '')).strip()
    symptoms_raw = str(item.get('symptoms_extract', '')).strip()
    
    if not disease or not symptoms_raw or symptoms_raw.lower() == "không có":
        continue
        
    symptoms = [s.strip() for s in symptoms_raw.replace('\\n', '\n').split('\n') if s.strip()]
    for s in symptoms:
        all_triples.append((disease, s))

# Loại bỏ trùng lặp và Lấy mẫu ngẫu nhiên
unique_triples = list(set(all_triples))
print(f"Tổng số Triples hợp lệ: {len(unique_triples)}")

# Lấy 300 mẫu ngẫu nhiên (Seed cố định để kết quả không bị đổi nếu chạy lại code)
random.seed(42) 
sample_size = min(300, len(unique_triples))
sampled_triples = random.sample(unique_triples, sample_size)

# Tạo DataFrame chuẩn bị xuất Excel
df_task = pd.DataFrame(sampled_triples, columns=['Ten_Benh', 'Trieu_Chung_Duoc_Trich_Xuat'])

# Tạo các cột trống để chuyên gia điền vào (1 = Đúng, 0 = Sai)
df_task['ChuyenGia_A'] = ""
df_task['ChuyenGia_B'] = ""
df_task['ChuyenGia_C'] = ""
df_task['Ghi_chu'] = ""

# Xuất ra file Excel
output_file = 'Real_Annotation_Task.xlsx'
df_task.to_excel(output_file, index=False)
print(f"Đã xuất thành công {sample_size} mẫu ra file: {output_file}")
print("=> BƯỚC TIẾP THEO: Hãy gửi file này cho các chuyên gia để họ điền 1 hoặc 0 vào các cột.")
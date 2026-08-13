import pandas as pd
import re

# 1. Đọc file bị lỗi (Bỏ qua dòng đầu tiên vì file của bạn có 2 dòng Header)
corrupted_df = pd.read_csv('neo4j_query_table_data_manual.csv', sep=';', skiprows=1, dtype=str)
corrupted_df.columns = ['Ma_ICD10', 'Ten_Benh', 'Quan_He', 'Trieu_Chung']

# 2. Đọc file dữ liệu sạch làm "Từ điển" khôi phục
clean_df = pd.read_excel('TA_VM_cleaned.xlsx', sep=';')
clean_disease_to_symptoms = {row['Tên bệnh']: row['Triệu chứng'] for _, row in clean_df.dropna(subset=['Tên bệnh']).iterrows()}
clean_disease_names = list(clean_disease_to_symptoms.keys())

def recover_text(corrupted_text, search_space):
    if pd.isna(corrupted_text) or not search_space: 
        return corrupted_text
    
    # Biến dấu '?' thành dấu '.' trong Regex để đại diện cho bất kỳ ký tự nào bị mất
    escaped_pattern = re.escape(corrupted_text).replace(r'\?', '.')
    
    # Khôi phục Tên Bệnh
    if isinstance(search_space, list):
        for clean_item in search_space:
            if re.match('^' + escaped_pattern + '$', str(clean_item), re.IGNORECASE):
                return clean_item
        return corrupted_text
    
    # Khôi phục Triệu chứng
    elif isinstance(search_space, str):
        match = re.search(escaped_pattern, str(search_space), re.IGNORECASE)
        return match.group(0) if match else corrupted_text

print("Đang khôi phục dữ liệu...")

# 3. Tiến hành khôi phục từng dòng
for idx, row in corrupted_df.iterrows():
    c_name = row['Ten_Benh']
    if pd.isna(c_name): continue
    
    # Phục hồi tên bệnh
    fixed_name = recover_text(c_name, clean_disease_names)
    corrupted_df.at[idx, 'Ten_Benh'] = fixed_name
    
    # Phục hồi triệu chứng
    c_symptoms_text = row['Trieu_Chung']
    if pd.notna(c_symptoms_text):
        c_symptoms = c_symptoms_text.split('\n')
        clean_block = clean_disease_to_symptoms.get(fixed_name, "")
        
        fixed_symptoms = []
        for s in c_symptoms:
            fixed_s = recover_text(s, clean_block)
            fixed_symptoms.append(fixed_s)
            
        corrupted_df.at[idx, 'Trieu_Chung'] = '\n'.join(fixed_symptoms)

# 4. Lưu lại file đã khôi phục với Encoding chuẩn (utf-8-sig)
output_file = 'Fixed_neo4j_query_table_data.csv'
corrupted_df.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')

print(f"Đã khôi phục xong! Dữ liệu được lưu tại: {output_file}")
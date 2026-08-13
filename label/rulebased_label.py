import pandas as pd
import json
import re

def clean_text(text):
    """Làm sạch văn bản: viết thường và loại bỏ ký tự đặc biệt."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

def evaluate_diseases(csv_path, json_path, output_path):
    # 1. Tải dữ liệu
    with open(json_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    
    # Giả định test.csv sử dụng dấu phân cách là ';' dựa trên cấu trúc file của bạn
    df_test = pd.read_csv(csv_path, sep=';')
    
    # Danh sách để lưu kết quả cuối cùng
    final_results = []

    print(f"Đang xử lý {len(df_test)} dòng dữ liệu từ file test...")

    # 2. Duyệt qua từng dòng trong file test.csv
    for index, row in df_test.iterrows():
        user_query = row['query']
        clean_query = clean_text(user_query)
        
        match_scores = []

        # 3. Đối chiếu với từng bệnh trong ICD10_LLM.json
        for disease in knowledge_base:
            disease_name = disease.get('ten_benh', 'Không rõ')
            # Lấy danh sách triệu chứng và chuẩn hóa
            raw_symptoms = disease.get('symptoms_extract', '')
            # Triệu chứng thường được phân tách bằng dấu xuống dòng hoặc dấu phẩy
            symptoms_list = [s.strip().lower() for s in raw_symptoms.replace('\n', ',').split(',') if s.strip()]
            
            score = 0
            matched_symptoms = []
            
            # Tính điểm dựa trên số lượng từ khóa triệu chứng xuất hiện trong query
            for sym in symptoms_list:
                if sym in clean_query:
                    score += 1
                    matched_symptoms.append(sym)
            
            if score > 0:
                match_scores.append({
                    'ten_benh': disease_name,
                    'score': score
                })

        # 4. Sắp xếp và lấy Top 1
        # Ưu tiên điểm cao trước, nếu bằng điểm thì ưu tiên tên bệnh (alphabet)
        top_1 = sorted(match_scores, key=lambda x: (-x['score'], x['ten_benh']))[:1]
        
        # Format kết quả nhãn duy nhất
        label = top_1[0]['ten_benh'] if top_1 else "Không rõ"
        
        # Thêm vào danh sách kết quả
        final_results.append({
            'query': user_query,
            'label': label
        })

    # 5. Xuất kết quả ra file mới
    output_df = pd.DataFrame(final_results)
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Hoàn thành! Kết quả đã được lưu tại: {output_path}")

# --- Thực thi ---
if __name__ == "__main__":
    CSV_INPUT = 'labeled/test.csv'
    JSON_INPUT = 'data/json/ICD10_LLM.json'
    CSV_OUTPUT = 'rulebased_labeled.csv'
    
    evaluate_diseases(CSV_INPUT, JSON_INPUT, CSV_OUTPUT)
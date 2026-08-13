import pandas as pd

def evaluate_similarity(file_llm, file_rulebased):
    # 1. Đọc dữ liệu
    df_llm = pd.read_csv(f"data/cleaned/{file_llm}")
    df_rule = pd.read_csv(f"data/cleaned/{file_rulebased}")

    # 2. Chuẩn hóa tên bệnh để merge (tránh lỗi viết hoa/thường, dư khoảng trắng)
    df_llm['ten_benh'] = df_llm['ten_benh'].astype(str).str.strip().str.lower()
    df_rule['ten_benh'] = df_rule['ten_benh'].astype(str).str.strip().str.lower()

    # 3. Gộp (Merge) dữ liệu theo tên bệnh
    df_merged = pd.merge(df_rule[['ten_benh', 'symptoms_extract']], 
                         df_llm[['ten_benh', 'symptoms_extract']], 
                         on='ten_benh', how='inner', suffixes=('_rule', '_llm'))

    total_records = len(df_merged)
    
    if total_records == 0:
        print("❌ Không có bệnh nào khớp giữa 2 file.")
        return

    jaccard_scores = []
    exact_matches = 0
    
    # 4. Duyệt qua từng bệnh để đối chiếu
    for index, row in df_merged.iterrows():
        # Lấy text triệu chứng (Xử lý trường hợp dữ liệu NaN/Null)
        symp_rule = str(row['symptoms_extract_rule']).lower() if pd.notna(row['symptoms_extract_rule']) else ""
        symp_llm = str(row['symptoms_extract_llm']).lower() if pd.notna(row['symptoms_extract_llm']) else ""
        
        # Tạo tập hợp (set) các triệu chứng, cắt theo dấu xuống dòng \n
        set_rule = set([s.strip() for s in symp_rule.split('\n') if s.strip()])
        set_llm = set([s.strip() for s in symp_llm.split('\n') if s.strip()])
        
        # Tính Tỷ lệ khớp hoàn toàn (Exact match)
        if set_rule == set_llm:
            exact_matches += 1
            
        # Tính Độ tương đồng Jaccard (Phần Giao chia cho Phần Hợp)
        intersection = len(set_rule.intersection(set_llm))
        union = len(set_rule.union(set_llm))
        
        if union > 0:
            jaccard = intersection / union
        else:
            # Nếu cả 2 hệ thống đều không tìm được gì thì coi như giống nhau 100%
            jaccard = 1.0 if len(set_rule) == 0 and len(set_llm) == 0 else 0.0
            
        jaccard_scores.append(jaccard)
        
    # 5. Tính trung bình toàn cục
    avg_jaccard = sum(jaccard_scores) / total_records
    accuracy = exact_matches / total_records
    
    # 6. In kết quả
    print(f"Tổng số bệnh so sánh               : {total_records}")
    print(f"Tỷ lệ khớp chính xác hoàn toàn 100%: {accuracy:.2%}")
    print(f"Độ tương đồng trung bình (Jaccard) : {avg_jaccard:.2%}")

# Chạy script
if __name__ == "__main__":
    evaluate_similarity('data_with_LLM.csv', 'data_with_rulebased.csv')
import pandas as pd

def calculate_agreement_final(file_path):
    print(f"Đang phân tích file: {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    def get_list(label_str):
        if pd.isna(label_str): return []
        return [item.strip().lower() for item in str(label_str).split(',')]

    total_rows = len(df)
    top1_exact_match = 0
    top1_rule_in_llm_top5 = 0
    jaccard_scores = []
    overlap_counts = []

    for _, row in df.iterrows():
        list_rule = get_list(row.get('label_rule', ''))
        list_llm = get_list(row.get('label_llm', ''))
        
        set_rule = set(list_rule)
        set_llm = set(list_llm)
        
        # A. Kiểm tra khớp chính xác Top-1
        if list_rule and list_llm and list_rule[0] == list_llm[0]:
            top1_exact_match += 1
        
        # B. Kiểm tra Top-1 của Rule có nằm trong Top-5 của LLM không
        if list_rule and list_rule[0] in set_llm:
            top1_rule_in_llm_top5 += 1
            
        # C. Tính chỉ số Jaccard (Độ tương đồng tập hợp)
        intersection = set_rule.intersection(set_llm)
        union = set_rule.union(set_llm)
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0
        jaccard_scores.append(jaccard)
        overlap_counts.append(len(intersection))

    # 4. Tính toán kết quả tổng hợp
    avg_jaccard = sum(jaccard_scores) / total_rows
    avg_overlap = sum(overlap_counts) / total_rows
    top1_exact_rate = (top1_exact_match / total_rows) * 100
    top1_in_top5_rate = (top1_rule_in_llm_top5 / total_rows) * 100

    # 5. Hiển thị báo cáo
    print(f"Tổng số mẫu:                       {total_rows}")
    print(f"Khớp chính xác Top-1:              {top1_exact_rate:.2f}%")
    print(f"LLM đồng ý với Top-1 của Rule:      {top1_in_top5_rate:.2f}%")
    print(f"Số bệnh trùng khớp trung bình:     {avg_overlap:.2f} / 5")
    print(f"Chỉ số Jaccard Similarity:         {avg_jaccard:.4f}")

if __name__ == "__main__":
    # Đảm bảo file final.csv nằm cùng thư mục với script này
    FILE_NAME = 'data/labeled/label_final.csv'
    calculate_agreement_final(FILE_NAME)
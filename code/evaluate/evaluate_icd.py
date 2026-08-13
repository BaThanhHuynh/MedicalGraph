import pandas as pd

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN FILE
# ==========================================
# Cập nhật đường dẫn đúng với môi trường của bạn (ví dụ: trên Colab)
FILE_RULE_BASED = "data/cleaned/ICD10_embed.csv" 
FILE_LLM = "data/cleaned/ICD10_LLM.csv"          
OUTPUT_CONFLICT_FILE = "ICD10_Conflicts_For_Manual_Check.csv"

def evaluate_icd_mapping():
    try:
        # ==========================================
        # 2. ĐỌC DỮ LIỆU
        # ==========================================
        df_rule = pd.read_csv(FILE_RULE_BASED)
        df_llm = pd.read_csv(FILE_LLM)

        # Chuẩn hóa tên bệnh để dùng làm khóa kết nối (key)
        df_rule['ten_benh'] = df_rule['ten_benh'].astype(str).str.strip().str.lower()
        df_llm['ten_benh'] = df_llm['ten_benh'].astype(str).str.strip().str.lower()

        # Giữ lại các cột cần thiết trước khi gộp
        df_rule = df_rule[['ten_benh', 'icd_code', 'icd_name']]
        df_llm = df_llm[['ten_benh', 'icd_code', 'icd_name']]

        # Đổi tên cột để dễ phân biệt
        df_rule = df_rule.rename(columns={'icd_code': 'icd_code_rule', 'icd_name': 'icd_name_rule'})
        df_llm = df_llm.rename(columns={'icd_code': 'icd_code_llm', 'icd_name': 'icd_name_llm'})

        # ==========================================
        # 3. GỘP VÀ PHÂN TÍCH DỮ LIỆU (INNER JOIN)
        # ==========================================
        df_merged = pd.merge(df_rule, df_llm, on='ten_benh', how='inner')
        total_records = len(df_merged)

        if total_records == 0:
            print("❌ Lỗi: Không tìm thấy bệnh nào khớp nhau giữa 2 file để so sánh.")
            return

        # Tính toán độ tương đồng (Agreement Rate)
        # Kiểm tra xem mã ICD của 2 phương pháp có giống nhau hoàn toàn không
        df_merged['is_match'] = df_merged['icd_code_rule'] == df_merged['icd_code_llm']
        
        matched_count = df_merged['is_match'].sum()
        conflict_count = total_records - matched_count
        agreement_rate = matched_count / total_records

        # ==========================================
        # 4. IN BÁO CÁO THỐNG KÊ
        # ==========================================
        print(f"Tổng số bệnh đã so sánh  : {total_records}")
        print(f"Số mã giống : {matched_count} ca")
        print(f"Số mã khác    : {conflict_count} ca")
        print(f"Tỷ lệ đồng thuận: {agreement_rate:.2%}")

        # ==========================================
        # 5. TRÍCH XUẤT FILE XUNG ĐỘT CHO CHUYÊN GIA
        # ==========================================
        if conflict_count > 0:
            # Lọc ra những ca cãi nhau
            df_conflicts = df_merged[df_merged['is_match'] == False].copy()
            
            # Thêm các cột trống để chuyên gia đánh giá thủ công
            df_conflicts['chuyen_gia_chon_ben_nao'] = "" # Điền: 'Rule', 'LLM', hoặc 'Khac'
            df_conflicts['ma_dung_tuyet_doi'] = ""
            df_conflicts['ghi_chu'] = ""

            # Sắp xếp lại thứ tự cột cho đẹp mắt
            cols_order = [
                'ten_benh', 
                'icd_code_rule', 'icd_name_rule', 
                'icd_code_llm', 'icd_name_llm',
                'chuyen_gia_chon_ben_nao', 'ma_dung_tuyet_doi', 'ghi_chu'
            ]
            df_conflicts = df_conflicts[cols_order]

            # Lưu ra file CSV mới
            df_conflicts.to_csv(OUTPUT_CONFLICT_FILE, index=False, encoding='utf-8-sig')
            print(f"\n✅ Đã xuất {conflict_count} ca xung đột ra file: '{OUTPUT_CONFLICT_FILE}'")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra trong quá trình chạy: {e}")

# Chạy script
if __name__ == "__main__":
    evaluate_icd_mapping()
import pandas as pd
import re

def clean_icd10_refined(input_path, output_path):
    try:
        # 1. Đọc file với cấu hình mạnh mẽ để tránh lỗi Parser và FileNotFoundError
        # sep=None giúp tự nhận diện dấu phẩy hoặc chấm phẩy
        df = pd.read_csv(input_path, sep=None, engine='python', on_bad_lines='skip')
        
        # Đặt lại tên cột cho đồng nhất
        df.columns = ['icd_code', 'icd_name']
        
        # 2. Làm sạch cột mã ICD (icd_code)
        # Giữ nguyên IN HOA, xoá khoảng trắng, chỉ giữ lại chữ cái, số và dấu chấm (.)
        def clean_code(code):
            code = str(code).upper().strip()
            # Loại bỏ mọi ký tự không phải chữ cái, số hoặc dấu chấm
            code = re.sub(r'[^A-Z0-9\.]', '', code)
            return code
            
        df['icd_code'] = df['icd_code'].apply(clean_code)
        
        # 3. Làm sạch cột tên bệnh (icd_name)
        # Chuyển về chữ thường, xoá khoảng trắng thừa, xoá kí tự đặc biệt/dấu câu
        def clean_name(name):
            if pd.isna(name): return ""
            name = str(name).lower().strip()
            # Loại bỏ dấu câu và kí tự đặc biệt (chỉ giữ chữ cái, số và khoảng trắng)
            # Re này giữ lại các kí tự Unicode (bao gồm cả tiếng Việt)
            name = re.sub(r'[^\w\s]', ' ', name)
            # Xoá khoảng trắng thừa ở giữa các từ
            name = re.sub(r'\s+', ' ', name).strip()
            return name
            
        df['icd_name'] = df['icd_name'].apply(clean_name)
        
        # 4. Loại bỏ các dòng bị trống sau khi làm sạch
        df = df[df['icd_code'] != '']
        df = df[df['icd_name'] != '']
        
        # 5. Xoá bỏ trùng lặp mã ICD (Chỉ giữ lại dòng đầu tiên của mỗi mã)
        count_before = len(df)
        df = df.drop_duplicates(subset=['icd_code'], keep='first')
        count_after = len(df)
        
        # 6. Lưu file kết quả
        # Dùng utf-8-sig để Excel hiển thị đúng tiếng Việt
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"--- BÁO CÁO LÀM SẠCH ---")
        print(f"Số lượng dòng ban đầu: {count_before}")
        print(f"Số lượng dòng trùng lặp đã xoá: {count_before - count_after}")
        print(f"Số lượng dòng cuối cùng: {count_after}")
        print(f"File kết quả đã được lưu tại: {output_path}")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Cấu hình file (Hãy kiểm tra đúng tên file trong thư mục của bạn)
INPUT_FILE = "data/ICD10_Reference.csv"
OUTPUT_FILE = "ICD10_Cleaned_Final.csv"

if __name__ == "__main__":
    clean_icd10_refined(INPUT_FILE, OUTPUT_FILE)
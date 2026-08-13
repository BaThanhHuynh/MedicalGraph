import pandas as pd
import re
import json
import time
import os

# ==========================================
# 1. TỪ ĐIỂN Y KHOA (RULE-BASED DICTIONARY)
# ==========================================
EXACT_SYMPTOMS = [
    # --- Triệu chứng toàn thân & Sinh hiệu ---
    "sốt", "ớn lạnh", "lạnh run", "rùng mình", "vã mồ hôi", "đổ mồ hôi", "mệt mỏi", "suy nhược", 
    "kiệt sức", "sụt cân", "giảm cân", "tăng cân", "béo phì", "chán ăn", "ăn không ngon", "khát nước",
    
    # --- Tiêu hóa & Bài tiết ---
    "buồn nôn", "nôn", "tiêu chảy", "táo bón", "đầy hơi", "chướng bụng", "khó tiêu", "ợ hơi", 
    "ợ chua", "trào ngược", "nấc cụt", "són tiểu", "vô niệu", "trĩ", "lòi dom",
    
    # --- Hô hấp & Tim mạch ---
    "ho", "khó thở", "thở dốc", "thở khò khè", "ngạt thở", "sổ mũi", "nghẹt mũi", "hắt hơi", 
    "hồi hộp", "đánh trống ngực",
    
    # --- Thần kinh & Tâm lý ---
    "chóng mặt", "hoa mắt", "co giật", "động kinh", "giật mình", "ngất xỉu", "hôn mê", 
    "lú lẫn", "đãng trí", "mất ngủ", "trầm cảm", "căng thẳng", "bứt rứt", "cáu gắt", 
    "ảo giác", "mê sảng", "hoang tưởng",
    
    # --- Tai - Mũi - Họng & Mắt ---
    "ù tai", "nói ngọng", "câm", "mù", "nhạt miệng", "đắng miệng", "khô miệng",
    
    # --- Da liễu, Cơ xương khớp & Phụ/Nam khoa ---
    "phát ban", "mề đay", "mẩn ngứa", "mụn nhọt", "bong tróc", "vàng da", "vàng mắt", "nhợt nhạt", 
    "bầm tím", "rụng tóc", "hói đầu", "loãng xương", "vô kinh", "rong kinh", "sẩy thai"
]

SYMPTOM_PREFIXES = [
    # --- Cảm giác & Đau đớn ---
    "đau", "nhức", "ngứa", "tức", "khó", "tê", "buốt", "rát", "xót", "mỏi", "châm chích", 
    "cảm giác", "hay bị", "chướng",
    
    # --- Viêm nhiễm & Tổn thương hình thái ---
    "sưng", "viêm", "tấy", "lở", "loét", "nổi", "u", "hạch", "nang", "polyp", "nhiễm", 
    "phù", "phù nề", "teo", "phình", "giãn", "xơ", "cứng", "mềm", "khuyết", "rách", 
    "gãy", "chệch", "trật", "vẹo", "nhô", "bầm",
    
    # --- Chức năng bị ảnh hưởng ---
    "mất", "giảm", "tăng", "rối loạn", "liệt", "suy", "yếu", "chậm", "nhanh", 
    "mờ", "điếc", "nghẹn", "co thắt", "co cứng", "căng",
    
    # --- Bài tiết, xuất tiết (Các tiền tố rất hay gặp ghép với danh từ) ---
    "chảy", "chảy máu", "xuất huyết", "ho ra", "khạc", "khạc ra", "nôn ra", 
    "đi ngoài", "đi cầu", "đi tiêu", "tiểu", "đái", "xuất tinh"
]

# Thêm các từ mang nghĩa phủ định lâm sàng
NEGATION_WORDS = [
    "không", "chưa", "chẳng", "không có", "chưa từng", "hiếm khi", "hầu như không", 
    "vắng mặt", "âm tính", "loại trừ", "chưa ghi nhận", "không thấy", "không phát hiện", 
    "hết", "biến mất", "không còn"
]

# Regex bắt các cụm từ chỉ thời gian và tần suất
TIME_PATTERNS = [
    r"kéo dài\s+(?:\w+\s+){1,4}(?:ngày|tuần|tháng|năm|giờ|phút)",
    r"(?:vài|một số|những)\s+(?:ngày|tuần|tháng|năm)",
    r"(?:thường xuyên|đôi khi|thỉnh thoảng|liên tục|đột ngột|về đêm|ban đêm|ban ngày)",
    r"(?:sau khi|trước khi|trong khi)\s+(?:\w+\s+){1,3}",
    r"\d+\s*(?:-\s*\d+\s*)?(?:ngày|tuần|tháng|năm)"
]

# ==========================================
# 2. HÀM RULE-BASED EXTRACTION
# ==========================================
def extract_entities_rule_based(text):
    if pd.isna(text) or str(text).strip() == "":
        return {"symptoms": ["Không có"], "negated_symptoms": ["Không có"], "duration": "Không có"}

    text = str(text).lower().strip()
    
    extracted_symptoms = set()
    negated_symptoms = set()
    extracted_durations = set()
    
    # Bước 1: Quét Thời gian/Tần suất bằng Regex
    for pattern in TIME_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            extracted_durations.add(match.strip())

    # Bước 2: Tách mệnh đề để xử lý Phủ định (Negation)
    clauses = re.split(r'[,.;:\n\(\)]', text)
    
    for clause in clauses:
        clause = clause.strip()
        if not clause: continue
        
        # Bắt từ phủ định trong mệnh đề hiện tại
        is_negated = any(clause.startswith(neg) or f" {neg} " in f" {clause} " for neg in NEGATION_WORDS)
        
        clause_findings = []
        
        # 2A: Quét khớp từ chính xác (Exact match)
        for sym in EXACT_SYMPTOMS:
            if re.search(r'\b' + sym + r'\b', clause):
                clause_findings.append(sym)
                
        # 2B: Quét cụm từ qua Tiền tố (Prefix match)
        for pref in SYMPTOM_PREFIXES:
            # Lấy tiền tố + tối đa 5 từ đi kèm phía sau
            pattern = rf"\b({pref}\s+(?:\w+\s*){{1,5}})"
            matches = re.findall(pattern, clause)
            
            for match in matches:
                phrase = match.strip()
                # Cắt bỏ các từ nối thừa ở đuôi cụm từ
                phrase = re.sub(r'\s+(và|hoặc|hay|của|thì|là|do)$', '', phrase).strip()
                if len(phrase.split()) > 1:
                    clause_findings.append(phrase)
        
        # Đưa vào danh sách Khẳng định hoặc Phủ định
        if is_negated:
            negated_symptoms.update(clause_findings)
        else:
            extracted_symptoms.update(clause_findings)
            
    # Xử lý kết quả trả về
    final_symptoms = list(extracted_symptoms) if extracted_symptoms else ["Không có"]
    final_negated = list(negated_symptoms) if negated_symptoms else ["Không có"]
    final_duration = ", ".join(list(extracted_durations)) if extracted_durations else "Không có"
            
    return {
        "symptoms": final_symptoms,
        "negated_symptoms": final_negated,
        "duration": final_duration
    }

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH (MAIN PROCESS)
# ==========================================
if __name__ == "__main__":
    # Đảm bảo tên file này khớp với file CSV bạn để trong thư mục dự án
    INPUT_FILE = "CleanMedical_data.xlsx" 
    OUTPUT_FILE = "RuleBased_Extracted_Symptoms.json"

    print("--- BẮT ĐẦU TRÍCH XUẤT RULE-BASED ---")
    
    # Kiểm tra xem file có tồn tại trong thư mục hiện tại không
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Lỗi: Không tìm thấy file '{INPUT_FILE}' trong thư mục hiện tại.")
        print(f"📁 Thư mục đang chạy code: {os.getcwd()}")
        print("👉 Vui lòng copy file dữ liệu vào thư mục này hoặc đổi lại đường dẫn INPUT_FILE.")
    else:
        try:
            # Đọc file CSV
            df = pd.read_excel(INPUT_FILE)
            
            # GIỚI HẠN DÒNG TEST: Đang để chạy 5 dòng đầu tiên. 
            # Xóa dòng dưới đây hoặc thay đổi số 5 nếu muốn chạy toàn bộ dữ liệu!
            
            results = []
            print(f"Tổng số dòng cần xử lý: {len(df)}\n")

            start_total = time.time()

            for index, row in df.iterrows():
                ten_benh = str(row.get("Tên bệnh", f"Dòng {index+1}")).strip()
                trieu_chung_raw = str(row.get("Triệu chứng", ""))
                
                print(f"[{index + 1}/{len(df)}] Đang xử lý: {ten_benh} ...", end=" ")
                
                # Chạy hàm Rule-based
                extracted = extract_entities_rule_based(trieu_chung_raw)
                
                print("✅ Xong!")
                print("    ↳ Triệu chứng:", ", ".join(extracted["symptoms"]))
                print("    ↳ Phủ định:", ", ".join(extracted["negated_symptoms"]))
                print("    ↳ Thời gian:", extracted["duration"])
                print("-" * 40)
                
                results.append({
                    "ten_benh": ten_benh,
                    "trieu_chung_goc": trieu_chung_raw,
                    "symptoms_extract": "\n".join(extracted["symptoms"]),
                    "negated_symptoms": "\n".join(extracted["negated_symptoms"]),
                    "duration": extracted["duration"]
                })

            # Lưu kết quả
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            end_total = time.time()
            print(f"\n✅ TRÍCH XUẤT HOÀN TẤT SAU {end_total - start_total:.2f} GIÂY!")
            print(f"📁 File kết quả đã lưu tại: {os.path.abspath(OUTPUT_FILE)}")

        except Exception as e:
            print(f"\n❌ Lỗi hệ thống: {e}")
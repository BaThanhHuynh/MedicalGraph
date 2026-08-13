import os
import json
import re
import torch
from tqdm import tqdm
from transformers import pipeline

# =====================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & HẰNG SỐ
# =====================================
INPUT_PATH = 'RuleBased_Extracted_Symptoms.json' 
OUTPUT_PATH = 'Final_Hybrid_Extracted_Symptoms.json'
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Đưa System Prompt ra ngoài vòng lặp để tối ưu hiệu suất bộ nhớ
SYSTEM_PROMPT = """Bạn là một chuyên gia AI về bóc tách dữ liệu y khoa (Medical NLP). Nhiệm vụ của bạn là đọc [VĂN BẢN GỐC] và trích xuất TOÀN BỘ thông tin lâm sàng thành định dạng JSON chuẩn.

QUY TẮC TRÍCH XUẤT TUYỆT ĐỐI TUÂN THỦ:
1. 'symptoms' (Triệu chứng & Dấu hiệu thực thể): 
- Quét cạn kiệt và mở rộng phạm vi. Trích xuất cả triệu chứng thông thường (đau đầu, sốt...) LẪN các dấu hiệu hình thể/bất thường (VD: bàn chân đổ vào trong, gót chân vẹo ngoài, sờ thấy khối u, da nổi nốt đỏ...).
- Giữ độ dài cụm từ vừa đủ để mô tả trọn vẹn hình thái lâm sàng, nhưng phải CẮT BỎ các từ nối rườm rà (VD: "bệnh nhân xuất hiện tình trạng gót chân vẹo ngoài" -> chỉ trích xuất "gót chân vẹo ngoài").
- CỰC KỲ CẨN THẬN: Tuyệt đối không đưa các biểu hiện mang ý nghĩa phủ định vào danh sách này.

2. 'negated_symptoms' (Triệu chứng phủ định): 
- Nơi chứa các biểu hiện mà văn bản khẳng định là người bệnh KHÔNG MẮC PHẢI. 
- Dấu hiệu nhận biết: Đi kèm các từ "không", "chưa", "chẳng" (VD: "không ho", "không sốt", "chưa sưng nề").

3. 'duration' (Thời gian & Tần suất): 
- Gộp tất cả các mốc thời gian, khoảng thời gian mắc bệnh, hoặc tần suất (VD: "3 ngày nay", "về đêm", "thỉnh thoảng"). 
- Nếu có nhiều mốc, hãy nối chúng lại bằng dấu phẩy.

QUY TẮC ĐỊNH DẠNG ĐẦU RA:
- Trả về DUY NHẤT một mã JSON hợp lệ. KHÔNG giải thích, KHÔNG viết thành câu, KHÔNG sinh ra bất kỳ văn bản nào bên ngoài dấu `{}`.
- Nếu một trường hoàn toàn không có thông tin trong văn bản, bắt buộc trả về ["Không có"] (đối với mảng) hoặc "Không có" (đối với chuỗi thời gian).

[VÍ DỤ MẪU BẮT BUỘC TUÂN THỦ]
Văn bản: "Bệnh nhân tỉnh táo, không sốt, chân tay chưa sưng nề. Tuy nhiên than phiền hay bị giật mình, da nổi nốt đỏ về đêm, và quan sát thấy bàn chân đổ vào trong kéo dài 3 ngày nay."
JSON ĐẦU RA:
{
"symptoms": ["hay bị giật mình", "da nổi nốt đỏ", "bàn chân đổ vào trong"],
"negated_symptoms": ["sốt", "chân tay sưng nề"],
"duration": "về đêm, kéo dài 3 ngày nay"
}"""

# =====================================
# 2. NẠP DỮ LIỆU & KHỞI TẠO PIPELINE
# =====================================
print("--- BƯỚC 1: NẠP DỮ LIỆU ---")
if os.path.exists(INPUT_PATH):
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data_records = json.load(f)
    print(f"✅ Đã nạp dữ liệu: {len(data_records)} dòng bệnh án.")
else:
    raise FileNotFoundError(f"❌ Cảnh báo: Không tìm thấy file {INPUT_PATH}.")

print("\n--- BƯỚC 2: KHỞI TẠO PIPELINE QWEN 1.5B ---")
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("✅ Pipeline khởi tạo thành công!")

# =====================================
# 3. CÁC HÀM XỬ LÝ (HELPER FUNCTIONS)
# =====================================
def safe_json_load(text):
    """Hàm phụ trợ để trích xuất JSON an toàn từ câu trả lời của AI"""
    if not text: return None
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group()
        json_str = re.sub(r',\s*([\}\]])', r'\1', json_str) # Xóa dấu phẩy thừa
        try: 
            return json.loads(json_str)
        except Exception: 
            return None
    return None

def ask_qwen_extraction(original_text, draft_symptoms, draft_negated, draft_duration):
    user_content = (
        f"[VĂN BẢN GỐC]\n{original_text}\n\n"
        f"[GỢI Ý TỪ MÁY QUÉT (Rule-based)]\n"
        f"- Có: {draft_symptoms}\n"
        f"- Không bị: {draft_negated}\n"
        f"- Thời gian: {draft_duration}\n\n"
        f"[KẾT QUẢ JSON ĐẦU RA]\n"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = pipe(
        prompt, 
        max_new_tokens=400, 
        do_sample=False, 
        repetition_penalty=1.02,
        return_full_text=False 
    )
    
    ans = outputs[0]["generated_text"].strip()
    parsed_json = safe_json_load(ans)
    
    # Xử lý fallback nếu AI sinh lỗi JSON
    if parsed_json is None:
        return {
            "symptoms_extract": draft_symptoms, 
            "negated_symptoms": draft_negated, 
            "duration": draft_duration
        }

    sym_list = parsed_json.get("symptoms", [])
    neg_list = parsed_json.get("negated_symptoms", [])
    
    return {
        "symptoms_extract": "\n".join(sym_list) if isinstance(sym_list, list) and sym_list else "Không có",
        "negated_symptoms": "\n".join(neg_list) if isinstance(neg_list, list) and neg_list else "Không có",
        "duration": str(parsed_json.get("duration", "Không có")).strip()
    }

# =====================================
# 4. CHẠY VÒNG LẶP & AUTO-SAVE
# =====================================
print("\n--- BƯỚC 3: BẮT ĐẦU TRÍCH XUẤT ---")

# Kiểm tra file dở dang để resume
start_idx = 0
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
            start_idx = len(results)
            print(f"🔄 Tìm thấy dữ liệu cũ. Bắt đầu chạy tiếp từ dòng {start_idx}...")
    except json.JSONDecodeError:
        print("⚠️ File kết quả cũ bị lỗi. Chạy lại từ đầu.")
        results = []
else:
    results = []

for i in tqdm(range(start_idx, len(data_records)), desc="Tiến trình bóc tách"):
    item = data_records[i]
    
    # Lấy dữ liệu đầu vào, xử lý ký tự xuống dòng an toàn
    text = item.get("trieu_chung_goc", "")
    draft_sym = str(item.get("symptoms_extract", "Không có")).replace('\n', ', ')
    draft_neg = str(item.get("negated_symptoms", "Không có")).replace('\n', ', ')
    draft_dur = str(item.get("duration", "Không có"))
    
    try:
        # Gọi pipeline phân tích
        extracted_data = ask_qwen_extraction(text, draft_sym, draft_neg, draft_dur)
        
        # Cập nhật kết quả vào item
        item["symptoms_extract"] = extracted_data["symptoms_extract"]
        item["negated_symptoms"] = extracted_data["negated_symptoms"]
        item["duration"] = extracted_data["duration"]
        
    except Exception as e:
        print(f"\n⚠️ Lỗi ở dòng {i}: {e}. Giữ nguyên kết quả gốc.")
        # Nếu lỗi vẫn giữ nguyên bản nháp
        pass
        
    results.append(item)
    
    # Auto-save sau mỗi 10 dòng
    if i > 0 and i % 10 == 0:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, ensure_ascii=False, indent=4)

# Lưu lần cuối khi hoàn thành toàn bộ
with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
    json.dump(results, out_f, ensure_ascii=False, indent=4)

print("\n✅ HOÀN TẤT!")
print(f"📁 Dữ liệu đã được lưu tại: {os.path.abspath(OUTPUT_PATH)}")
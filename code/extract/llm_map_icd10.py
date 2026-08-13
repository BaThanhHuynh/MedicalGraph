import json
import os
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. CẤU HÌNH FILE DỮ LIỆU
# ==========================================
INPUT_FILE = "ICD10_embedding.json"
OUTPUT_FILE = "ICD10_LLM.json"

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH QWEN 2.5 3B
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("⏳ Đang tải mô hình Qwen2.5-3B (Sẽ tốn thời gian tải ở lần chạy đầu tiên)...")
# Tự động nạp model lên GPU nếu có, ngược lại sẽ dùng RAM/CPU
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto", 
    device_map="auto" 
)
print("✅ Tải mô hình thành công!\n")

# ==========================================
# 3. HÀM PHỤ TRỢ (XỬ LÝ LỖI LLM)
# ==========================================
def extract_json_from_text(text):
    """
    Dùng Regex để bóc tách JSON. 
    Phòng trường hợp mô hình sinh ra văn bản thừa.
    """
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {} # Trả về dict rỗng nếu lỗi để giữ nguyên mã cũ

# ==========================================
# 4. HÀM ĐÁNH GIÁ CHÍNH
# ==========================================
def evaluate_single_record(record):
    ten_benh = record.get("ten_benh", "Không rõ")
    
    # Dọn dẹp ký tự xuống dòng \n thành dấu phẩy để LLM dễ đọc hơn
    symptoms = record.get("symptoms_extract", "Không rõ")
    if isinstance(symptoms, str):
        symptoms = symptoms.replace('\n', ', ')
        
    old_icd_code = record.get("icd_code", "")
    old_icd_name = record.get("icd_name", "")

    # Prompt bám sát logic mới: Chỉ trả về mã chuẩn và tên chuẩn
    system_prompt = """Bạn là chuyên gia y khoa ICD-10 của WHO tại Việt Nam.
Nhiệm vụ: Kiểm tra xem mã ICD-10 đang gán cho bệnh nhân có đúng không.
- Nếu mã hệ thống gán ĐÃ ĐÚNG: Trả về chính xác mã và tên đó.
- Nếu mã hệ thống gán SAI: Hãy suy luận và tìm mã chuẩn xác nhất.
BẮT BUỘC chỉ trả về 1 chuỗi JSON duy nhất gồm 2 trường sau, không giải thích gì thêm:
{"icd_code": "Mã chuẩn", "icd_name": "Tên bệnh theo mã"}"""

    user_prompt = f"""Tên bệnh: {ten_benh}
Triệu chứng: {symptoms}
Mã ICD-10 đang gán: {old_icd_code} - {old_icd_name}

Kết quả JSON:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Chuẩn bị dữ liệu đầu vào cho mô hình
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    # Chạy mô hình sinh văn bản
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=150,
            temperature=0.1, # Nhiệt độ thấp giúp mô hình tuân thủ định dạng tốt hơn
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Lấy câu trả lời (loại bỏ phần prompt gốc)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return extract_json_from_text(response_text)

# ==========================================
# 5. CHẠY PIPELINE (TÍCH HỢP RESUME)
# ==========================================
def main():
    patient_data = []
    
    # LOGIC MỚI 1: Tải file đang chạy dở (nếu có) để chạy tiếp
    if os.path.exists(OUTPUT_FILE):
        print(f"Phát hiện file chạy dở '{OUTPUT_FILE}'. Đang tải dữ liệu để chạy tiếp...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            patient_data = json.load(f)
    else:
        if not os.path.exists(INPUT_FILE):
            print(f"❌ Lỗi: Không tìm thấy file gốc '{INPUT_FILE}'.")
            return
        print(f"Đang đọc dữ liệu mới từ '{INPUT_FILE}'...")
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            patient_data = json.load(f)

    print(f"Bắt đầu nhờ Qwen2.5-3B kiểm tra {len(patient_data)} bản ghi...\n")
    
    # Thanh tiến trình
    for i, record in enumerate(tqdm(patient_data, desc="Đang xử lý")):
        # Bỏ qua nếu dòng này đã được xử lý ở lần chạy trước
        if record.get('_is_checked') == True:
            continue
            
        old_icd_code = record.get('icd_code', '')
        
        # Gọi mô hình đánh giá
        evaluation_result = evaluate_single_record(record)
        
        # LOGIC MỚI 2: Ghi đè trực tiếp kết quả nếu nhận được JSON hợp lệ
        new_icd_code = evaluation_result.get("icd_code", old_icd_code)
        new_icd_name = evaluation_result.get("icd_name", record.get("icd_name", ""))
        
        record["icd_code"] = new_icd_code
        record["icd_name"] = new_icd_name
        record["_is_checked"] = True # Đánh dấu đã xong

        # In thông báo ra màn hình (tùy chọn, bạn có thể comment dòng print lại nếu thấy rối khi dùng tqdm)
        if new_icd_code != old_icd_code and new_icd_code:
             tqdm.write(f" ⚠️ Sửa: {record.get('ten_benh')} ({old_icd_code} -> {new_icd_code})")

        # Tự động lưu file mỗi 50 câu để chống mất dữ liệu
        if (i + 1) % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(patient_data, f, ensure_ascii=False, indent=4)

    # LOGIC MỚI 3: Dọn dẹp cờ _is_checked trước khi lưu file cuối cùng
    for item in patient_data:
        item.pop('_is_checked', None)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(patient_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Hoàn tất! Đã lưu kết quả đối soát tại file: '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
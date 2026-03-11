import os
import torch
import pandas as pd
import json
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

INPUT_FILE = "data/DataBase (1).xlsx"  
OUTPUT_FILE = "Structured_Extracted_Symptoms.json"
MAX_NEW_TOKENS = 400

print("--- BƯỚC 1: KIỂM TRA DỮ LIỆU ---")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {INPUT_FILE}\nVui lòng tạo thư mục 'data' và đặt file Excel vào đó.")
print(f"Đã nhận diện file: {INPUT_FILE}")

print("\n--- BƯỚC 2: KHỞI TẠO MÔ HÌNH AI ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
print(f"Model tải thành công lên thiết bị: {model.device}")

SYSTEM_PROMPT = """Bạn là chuyên gia bóc tách dữ liệu y tế (Medical NLP) phục vụ nghiên cứu khoa học. Yêu cầu ĐỘ CHÍNH XÁC TUYỆT ĐỐI.

Nhiệm vụ: Phân tích văn bản bệnh án và trích xuất thông tin vào định dạng JSON.

QUY TẮC NGHIÊM NGẶT ĐỐI VỚI "symptoms":
1. Bóc tách tối đa: Nếu một câu chứa nhiều triệu chứng nối với nhau bằng dấu phẩy (,), chữ "và", "hoặc", "hay", phải tách chúng thành các cụm từ riêng lẻ.
2. Loại bỏ nhiễu: TUYỆT ĐỐI BỎ các đoạn văn giải thích cơ chế, nguyên nhân, tiến triển bệnh.
3. Viết thường: Chuyển toàn bộ kết quả về chữ viết thường (lowercase).
4. Phủ định và Thời gian: Trích xuất triệu chứng bị phủ định vào "negated_symptoms", và thời gian vào "duration".

VÍ DỤ MẪU BẮT BUỘC TUÂN THỦ:
[VĂN BẢN TRÍCH XUẤT]
Những triệu chứng áp xe gan không xuất hiện ngay, nhưng khi xuất hiện sẽ là biểu hiện lâm sàng rất nghiêm trọng. Những triệu chứng nguy hiểm của bệnh áp xe gan gồm: (2)
Buồn nôn, nôn
Sụt cân, chán ăn
Vã nhiều mồ hôi
Vàng da
Sốt cao kèm theo rét run: sốt 39°C – 40°C trong giai đoạn cấp tính của bệnh, sau đó giảm xuống và kéo dài
Đau tức ở hạ sườn phải hoặc cảm giác căng tức nặng ở vùng hạ sườn phải: Đây là triệu chứng do gan to. Khi ổ áp xe to cấp tính, cơn đau sẽ lan đến vùng thượng vị hay toàn bộ vùng bụng
Người bệnh cảm thấy đau khi ấn kẽ sườn 11- 12

[KẾT QUẢ JSON ĐẦU RA]
{
  "symptoms": ["buồn nôn",
  "nôn",
  "sụt cân",
  "chán ăn",
  "vã nhiều mồ hôi",
  "vàng da",
  "sốt cao kèm theo rét run",
  "đau tức ở hạ sườn phải",
  "cảm giác căng tức nặng ở vùng hạ sườn phải",
  "đau khi ấn kẽ sườn 11-12"],
  "negated_symptoms": [],
  "duration": ""
}"""

def clean_list(items):
    if not isinstance(items, list): return []
    seen = set()
    cleaned = []
    for i in items:
        val = str(i).strip().lower()
        if val and val not in seen:
            seen.add(val)
            cleaned.append(val)
    return cleaned

def safe_json_load(text):
    if not text: return None
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None

def extract_entities(text):
    if pd.isna(text) or str(text).strip() == "":
        return {"symptoms": [], "negated_symptoms": [], "duration": ""}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Thực hiện trích xuất theo định dạng mẫu cho văn bản sau:\n[VĂN BẢN TRÍCH XUẤT]\n{str(text)}"}
    ]

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response_ids = outputs[0][len(inputs.input_ids[0]):]
    raw_output = tokenizer.decode(response_ids, skip_special_tokens=True)

    data = safe_json_load(raw_output)
    if data is None:
        return {"symptoms": [], "negated_symptoms": [], "duration": ""}

    return {
        "symptoms": clean_list(data.get("symptoms", [])),
        "negated_symptoms": clean_list(data.get("negated_symptoms", [])),
        "duration": str(data.get("duration", "")).strip()
    }

print("\n--- BƯỚC 3: BẮT ĐẦU TRÍCH XUẤT ---")
try:
    df = pd.read_excel(INPUT_FILE)
    results = []

    print(f"Tổng số dòng cần xử lý: {len(df)}\n")

    for index, row in df.iterrows():
        ten_benh = str(row.get("Tên bệnh", f"Dòng {index+1}")).strip()
        print(f"[{index + 1}/{len(df)}] Đang phân tích: {ten_benh} ...", end=" ")

        trieu_chung_raw = str(row.get("Triệu chứng", ""))

        if len(trieu_chung_raw) > 2500:
            trieu_chung_raw = trieu_chung_raw[:2500] + "... (đã cắt bớt)"

        start_time = time.time()

        try:
            extracted = extract_entities(trieu_chung_raw)
            end_time = time.time()

            symptoms_str = "\n".join(extracted["symptoms"])
            negated_str = "\n".join(extracted["negated_symptoms"])

            print(f"Xong ({end_time - start_time:.1f}s)")

            if extracted["symptoms"]:
                print("    ↳ Triệu chứng:")
                for sym in extracted["symptoms"]:
                    print(f"      - {sym}")
            if extracted["negated_symptoms"]:
                print("    ↳ Phủ định:")
                for n_sym in extracted["negated_symptoms"]:
                    print(f"      - {n_sym}")

        except Exception as row_error:
            print(f"Lỗi: {row_error}")
            symptoms_str, negated_str, extracted = "", "", {"duration": ""}

        results.append({
            "ten_benh": ten_benh,
            "trieu_chung_goc": trieu_chung_raw,
            "symptoms_extract": symptoms_str,
            "negated_symptoms": negated_str,
            "duration": extracted.get("duration", "")
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nTRÍCH XUẤT HOÀN TẤT!")
    print(f"File kết quả đã được lưu tại: {os.path.abspath(OUTPUT_FILE)}")

except Exception as e:
    print(f"\nĐã xảy ra lỗi hệ thống: {e}")
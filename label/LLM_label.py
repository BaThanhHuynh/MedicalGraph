import pandas as pd
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Cấu hình mô hình Qwen2.5-3B
model_name = "Qwen/Qwen2.5-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Đang tải mô hình {model_name} lên {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 2. Tải dữ liệu kiến thức (Knowledge Base)
with open('ICD10_LLM.json', 'r', encoding='utf-8') as f:
    kb = json.load(f)
disease_names = [d['ten_benh'] for d in kb]

# 3. Hàm dự đoán bằng LLM
def get_llm_labels(query, disease_context):
    prompt = f"""
Bạn là một chuyên gia y tế AI. Dựa trên mô tả triệu chứng của bệnh nhân, hãy chọn Top 5 bệnh phù hợp nhất từ danh sách gợi ý.

Mô tả bệnh nhân: "{query}"

Danh sách gợi ý: {disease_context}

Yêu cầu:
- Chỉ trả về tên 5 bệnh, cách nhau bằng dấu phẩy.
- Sắp xếp theo độ phù hợp giảm dần.
- Không giải thích thêm.
"""
    messages = [
        {"role": "system", "content": "Bạn là trợ lý y tế chuyên nghiệp."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=256, temperature=0.1)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.split("assistant\n")[-1].strip()

# 4. Xử lý file labeled_results.csv
df = pd.read_csv('rulebased_labeled.csv')
results = []

# Để demo, chúng ta lấy 50 bệnh đầu tiên làm ngữ cảnh để tránh tràn Token (Context Window)
context = ", ".join(disease_names[:100])

print("Bắt đầu đánh nhãn lại bằng LLM...")
for idx, row in df.iterrows():
    print(f"Đang xử lý dòng {idx+1}/300...")
    new_labels = get_llm_labels(row['query'], context)
    results.append({
        "query": row['query'],
        "rule_based_labels": row['predicted_labels'],
        "llm_refined_labels": new_labels
    })

# 5. Lưu kết quả
output_df = pd.DataFrame(results)
output_df.to_csv('llm_labeled.csv', index=False, encoding='utf-8-sig')

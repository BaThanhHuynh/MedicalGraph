import pandas as pd

df = pd.read_csv('label/final.csv')
if 'ground_truth' not in df.columns:
    df['ground_truth'] = None

for idx, row in df.iterrows():
    if pd.notna(row['ground_truth']): continue # Bỏ qua nếu đã có nhãn
    
    print(f"\n--- Câu hỏi {idx+1}/300 ---")
    print(f"Query: {row['query']}")
    print(f"1. Rule: {row['label_rule']}")
    print(f"2. LLM:  {row['label_llm']}")
    
    choice = input("Chọn 1, 2 hoặc gõ nhãn mới (S để lưu và thoát): ")
    
    if choice == '1':
        df.at[idx, 'ground_truth'] = row['label_rule']
    elif choice == '2':
        df.at[idx, 'ground_truth'] = row['label_llm']
    elif choice.lower() == 's':
        break
    else:
        df.at[idx, 'ground_truth'] = choice

df.to_csv('final_with_gt.csv', index=False, encoding='utf-8-sig')
import pandas as pd
import sys
import os
import time

# ==========================================
# 0. SỬA LỖI ĐƯỜNG DẪN IMPORT
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from Faiss.app import load_system, search as faiss_search
    from test_RAG import retrieve_diseases_via_graph as graphrag_search
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

# ==========================================
# 1. HÀM TÍNH TOÁN METRICS
# ==========================================
def calculate_metrics(expected_disease, predicted_list):
    if pd.isna(expected_disease):
        return 0, 0, 0.0
        
    expected = str(expected_disease).lower().strip()
    predictions = [str(p).lower().strip() for p in predicted_list]

    # Kiểm tra Hit@1: Có khớp ngay ở vị trí đầu tiên không?
    hit_1 = 1 if len(predictions) > 0 and expected in predictions[0] else 0
    
    # Kiểm tra Hit@5: Có chứa trong Top 5 bệnh không?
    hit_5 = 1 if any(expected in p for p in predictions) else 0
    
    # Tính MRR
    mrr = 0.0
    for rank, p in enumerate(predictions, 1):
        if expected in p:
            mrr = 1.0 / rank
            break
            
    return hit_1, hit_5, mrr

# ==========================================
# 2. CHẠY ĐÁNH GIÁ (EVALUATION RUN)
# ==========================================
def run_evaluation(csv_file_path):
    print(f"[*] Đang đọc dữ liệu từ file: {csv_file_path}...")
    try:
        df_test = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return

    # Kiểm tra xem file có đúng cột không
    if 'query' not in df_test.columns or 'predicted_labels' not in df_test.columns:
        print("❌ Lỗi: File phải có cột 'query' và 'predicted_labels'.")
        return

    total_cases = len(df_test)
    print(f"[*] Tổng số ca test: {total_cases}")
    
    print("[*] Đang khởi tạo hệ thống FAISS...")
    f_model, f_index, f_docs = load_system()
    
    print("[*] Bắt đầu đánh giá (Có thể mất vài phút)...\n")
    results = []
    
    for idx, row in df_test.iterrows():
        query = str(row['query'])
        
        # Lấy bệnh đầu tiên trong chuỗi 5 bệnh của LLM làm Nhãn chuẩn (Ground Truth)
        labels_str = str(row['predicted_labels'])
        expected = labels_str.split(',')[0].strip() if pd.notna(labels_str) else ""
        
        if not expected:
            continue
            
        # In tiến độ
        if (idx + 1) % 10 == 0:
            print(f"   ... Đã xử lý {idx + 1}/{total_cases} ca")
            
        # --- 1. Chạy FAISS ---
        faiss_raw = faiss_search(query, f_model, f_index, f_docs, k=5)
        # Bắt lỗi cấu trúc trả về
        if len(faiss_raw) > 0 and isinstance(faiss_raw[0], dict):
            faiss_preds = [item.get('disease', item.get('name', '')) for item in faiss_raw]
        else:
            faiss_preds = faiss_raw
            
        f_h1, f_h5, f_mrr = calculate_metrics(expected, faiss_preds)
        
        # --- 2. Chạy GraphRAG ---
        symptoms_list = [s.strip() for s in query.split(',')]
        graph_raw = graphrag_search(symptoms_list)
        # Bắt lỗi cấu trúc trả về
        if len(graph_raw) > 0 and isinstance(graph_raw[0], dict):
            graph_preds = [item.get('disease', item.get('name', '')) for item in graph_raw]
        else:
            graph_preds = graph_raw
            
        g_h1, g_h5, g_mrr = calculate_metrics(expected, graph_preds)
        
        # Lưu kết quả từng câu
        results.append({
            "Query": query[:40] + "...", 
            "Ground_Truth (LLM Top 1)": expected,
            "FAISS_Hit5": f_h5, "FAISS_Hit1": f_h1, "FAISS_MRR": f_mrr,
            "Graph_Hit5": g_h5, "Graph_Hit1": g_h1, "Graph_MRR": g_mrr
        })
        
        # Nghỉ nhẹ 0.05s để tránh nghẽn CPU
        time.sleep(0.05)
        
    # --- 3. TỔNG HỢP VÀ IN BÁO CÁO ---
    df_results = pd.DataFrame(results)
    
    # Xuất báo cáo ra file CSV
    output_file = "evaluation_llm_rag.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*75)
    print(" 🏆 KẾT QUẢ ĐÁNH GIÁ (DỰA TRÊN NHÃN CHUẨN LÀ TOP-1 LLM)")
    print("="*75)
    print(f"Tổng số mẫu phân tích: {len(df_results)} ca bệnh")
    print(f"File kết quả chi tiết từng câu đã lưu tại: {output_file}")
    
    print(f"\n1. Tỉ lệ dự đoán đúng trong Top 5 (Hit@5):")
    print(f"   - FAISS    : {df_results['FAISS_Hit5'].mean() * 100:.2f}%")
    print(f"   - GraphRAG : {df_results['Graph_Hit5'].mean() * 100:.2f}%")
    
    print(f"\n2. Độ chính xác tuyệt đối ở vị trí số 1 (Hit@1):")
    print(f"   - FAISS    : {df_results['FAISS_Hit1'].mean() * 100:.2f}%")
    print(f"   - GraphRAG : {df_results['Graph_Hit1'].mean() * 100:.2f}%")
    
    print(f"\n3. Chỉ số xếp hạng MRR (Mean Reciprocal Rank):")
    print(f"   - FAISS    : {df_results['FAISS_MRR'].mean():.4f}")
    print(f"   - GraphRAG : {df_results['Graph_MRR'].mean():.4f}")
    print("="*75)

if __name__ == "__main__":
    # Điền đúng đường dẫn tới file llm_labeled.csv của bạn
    # Ví dụ: "../data/labeled/llm_labeled.csv" nếu file ở thư mục data
    TEST_FILE_PATH = "data/labeled/llm_labeled.csv" 
    run_evaluation(TEST_FILE_PATH)
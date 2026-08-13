import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import numpy as np

print("Đang đọc file Đánh giá thực tế từ các chuyên gia...")
# Đọc file Excel đã hoàn thiện
try:
    df = pd.read_excel('Completed_Annotation.xlsx')
except FileNotFoundError:
    print("LỖI: Không tìm thấy file Completed_Annotation.xlsx. Hãy đảm bảo bạn đã điền dữ liệu.")
    exit()

# Đảm bảo dữ liệu là dạng số nguyên (xóa khoảng trắng hoặc lỗi gõ phím nếu có)
raters = ['ChuyenGia_A (Bạn)', 'ChuyenGia_B (Chuyên gia 1)', 'ChuyenGia_C (Chuyên gia 2)']
for col in raters:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

print(f"Đã tải thành công {len(df)} mẫu đánh giá.")
print("="*70)

# 1. Tính toán Mức độ đồng thuận tổng thể (Fleiss' Kappa cho 3 người)
ratings_matrix = df[raters].values
agg_ratings, _ = aggregate_raters(ratings_matrix)
fleiss_k = fleiss_kappa(agg_ratings)

print("1. ĐÁNH GIÁ ĐỘ ĐỒNG THUẬN (FLEISS' KAPPA)")
print(f"   ► Điểm số Fleiss' Kappa: {fleiss_k:.4f}")
if fleiss_k > 0.6:
    print("   ► Đánh giá: ĐẠT CHUẨN NGHIÊN CỨU KHOA HỌC (>0.6)")
else:
    print("   ► Đánh giá: Độ đồng thuận thấp, cần xem lại hướng dẫn gán nhãn.")
print("-" * 70)

# 2. Tạo File Tiêu Chuẩn Vàng Cuối Cùng (Gold Standard Resolution)
# Biểu quyết đa số: Nếu >= 2 người đồng ý là Đúng (1), thì kết luận cuối cùng là Đúng.
df['Tong_Vote'] = df[raters].sum(axis=1)
df['Ket_Luan_Cuoi_Cung'] = np.where(df['Tong_Vote'] >= 2, 1, 0)

# Thống kê kết quả
so_luong_dung = df[df['Ket_Luan_Cuoi_Cung'] == 1].shape[0]
so_luong_sai = df[df['Ket_Luan_Cuoi_Cung'] == 0].shape[0]
precision_estimate = so_luong_dung / len(df)

print("2. KẾT QUẢ BIỂU QUYẾT TẬP MẪU (MAJORITY VOTING)")
print(f"   - Số Triples được hội đồng xác nhận ĐÚNG : {so_luong_dung}")
print(f"   - Số Triples bị hội đồng loại bỏ (SAI)   : {so_luong_sai}")
print(f"   ► Ước tính Precision thực tế của hệ thống: {precision_estimate:.2%}")
print("="*70)

# Lưu lại kết quả để báo cáo
df.to_excel('Final_Resolved_Annotations.xlsx', index=False)
print("Đã lưu chi tiết quá trình biểu quyết vào file 'Final_Resolved_Annotations.xlsx'")
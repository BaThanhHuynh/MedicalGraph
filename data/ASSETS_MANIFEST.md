# Bản kê tài sản dữ liệu — MedicalGraph / MedKG-HRR

Lập ngày 13/08/2026. Mục đích: mọi tệp cần để tái lập bài báo v10 đều có mặt ở đây
kèm SHA-256, kể cả tệp cố tình không đưa vào git. Mất tệp nào sẽ phát hiện được ngay
thay vì phát hiện sau khi cài lại máy.

Kiểm tra toàn vẹn:

```bash
cd data && sha256sum -c ASSETS_MANIFEST.sha256
```

## Có theo dõi trong git

| Tệp | Kích thước | SHA-256 |
|---|---|---|
| `medical_data_cleaned.csv` | 4,1 MB | `6e6840b39ef13c5406c86f4e6530dde2c3dc89ae3356b33da7d61a51c283239a` |
| `test_300_paraphrased.csv` | 116 KB | `dde92c58dc6cf6874b624eb9279d1d6ed0fd3d0fd963dd63b9a16b23db88470c` |
| `ICD10_cleaned.csv` | 772 KB | `e6dd28f228829f811cf63c36f9b3d1a67d9b8d95458f60aa7a8a908211526b6b` |
| `medical_chunks_hierarchical.csv` | 24 MB | `bbdc395245f17d56817e4e0282b65130e88a1f8880c26ab98f03a2e8668a1f30` |

> ⚠ `medical_chunks_hierarchical.csv` được đưa vào git ngày 13/08/2026, **với điều
> kiện repo giữ private** (đã xác nhận `github.com/BaThanhHuynh/MedicalGraph` trả
> HTTP 404 khi truy cập ẩn danh). Tệp chứa văn bản y khoa nguồn từ
> tamanhhospital.vn và vinmec.com; Mục 8.1 của bài báo nêu rõ không phát hành văn
> bản thô. Muốn mở repo sang public thì phải gỡ tệp khỏi **toàn bộ lịch sử git**
> bằng `git filter-repo`, xóa ở commit mới nhất là không đủ.

## Không theo dõi trong git — vẫn bắt buộc để chạy

| Tệp | Kích thước | SHA-256 | Lý do loại trừ |
|---|---|---|---|
| `faiss_medical_local_phobert.index` | 25 MB | `d117e7e4bddae43cd2ffd71bd3f6c938a65c7236f21a56690f8410607d202418` | nhị phân lớn; git không diff được, mỗi lần dựng lại history phình thêm 25 MB |

## Mô hình — ngoài repo

| Đường dẫn | Ghi chú |
|---|---|
| `C:\BACKUP_RESEARCH_2026-08-11\models\phobert-medical-constractive` | `model.safetensors` = `6f57558285c6944bc4a39014e8a1b3099fa49a37ef5964dc191f45487fe8c829` |
| `D:\Download\Medical_NLP\phobert-medical-constractive` | bản thứ hai, trùng SHA-256 |

---

## Ràng buộc quan trọng

**`medical_chunks_hierarchical.csv` và `faiss_medical_local_phobert.index` phải đi
thành cặp.** Chỉ mục là `IndexFlatIP`, 768 chiều, `ntotal = 8.214`; CSV có đúng
8.214 dòng. `retrieve_rag()` tra ngược tên bệnh bằng `df_rag.iloc[idx]` với `idx`
lấy từ FAISS, nên chỉ cần một trong hai tệp bị dựng lại riêng là toàn bộ ánh xạ
lệch dòng — kết quả vẫn chạy nhưng sai âm thầm, không báo lỗi.

## Nguồn khôi phục

Toàn bộ tìm thấy trong `D:\Download\` ngày 13/08/2026:

- `D:\Download\Medical_NLP\` — cây thư mục làm việc of-record đầy đủ (chỉ mục,
  corpus, dữ liệu, mô hình, notebook `eval (2).ipynb` mà `medkgbert_dx.py` được
  tách module từ đó).
- `D:\Download\Medical_NLP_Hien-20260725T155457Z-1-001\` — bản tải từ Drive
  `hienhocit205@gmail.com` ngày 25/07/2026.

Đã đối chiếu SHA-256 giữa các bản: chỉ mục FAISS giống nhau ở cả ba vị trí;
`medkgbert_dx.py` giống nhau ở cả ba bản. Không có mơ hồ về bản nào là of-record.

**Lưu ý tên tệp:** trên Drive, corpus RAG mang tên `rag_chunks.csv`; trong
`Medical_NLP\` nó tên `medical_chunks_hierarchical.csv`. Hai tệp trùng SHA-256 —
cùng một tệp, hai tên. Bản đối soát ngày 12/08 từng liệt kê chúng như hai tài sản
riêng biệt; thực tế chỉ có một.

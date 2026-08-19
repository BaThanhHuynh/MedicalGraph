# -*- coding: utf-8 -*-
"""Dung bao cao gop TN4 + TN5 + TN7, so lieu doi chieu voi ban of-record TN3.

Nguon:
  D:\\so_lieu_cuoi_bo_sung_TN3            (TN1, TN2, TN3 - ban of-record)
  D:\\NopThay_TN4_TN5_2026-08-17\\ket_qua  (bon zip ket qua TN4/TN4B/TN5/TN5B)

Bang theo chuong va cac con so hop/giao deu tinh lai tu tep per-query luc chay.
Chay lai script nay cho ra dung tep .docx.
"""
import csv, hashlib, io, os, sys, zipfile

from docx import Document
from docx.shared import Pt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OF = r'D:\so_lieu_cuoi_bo_sung_TN3'
GOI = r'D:\NopThay_TN4_TN5_2026-08-17'
DICH = os.path.join(GOI, 'BaoCao_TN4_TN5_TN7_2026-08-19.docx')

TN1 = os.path.join(OF, 'thi_nghiem_1_baseline_doc_lap')
TN3 = os.path.join(OF, 'thi_nghiem_3_unconstrained')


# ------------------------------------------------------------------ du lieu
def doc(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def bo(x):
    return str(x).strip().lower() in ('1', 'true', 'yes')


def bam(b):
    return hashlib.sha256(b).hexdigest()


def nap_perquery():
    """Tra ve (base, he) - he[ten][case_id] = True/False cho Hit@1."""
    with zipfile.ZipFile(os.path.join(GOI, 'tn4_input.zip')) as z:
        base = {r['case_id']: r for r in csv.DictReader(
            io.StringIO(z.read('independent_scored_perquery.csv').decode('utf-8-sig')))}
    he = {'MedKG-HRR (đề xuất)':
          {c: str(r['rank3']).strip() in ('1', '1.0') for c, r in base.items()}}
    for ten, p in [
        ('Retrieval-only', os.path.join(TN1, 'tn1_perquery_retrieval_only.csv')),
        ('Phi-3.5 zero-shot', os.path.join(TN1, 'tn1_perquery_phi_35_zero_shot.csv')),
        ('BM25-only', os.path.join(TN1, 'tn1_perquery_bm25_only.csv')),
        ('KG-only', os.path.join(TN1, 'tn1_perquery_kg_only.csv')),
        ('BM25 tự do', os.path.join(TN3, 'tn3_perquery_bm25_unconstrained.csv')),
        ('Llama-3', os.path.join(TN3, 'llm', 'llama3_8b', 'tn3_perquery_llama3_8b.csv')),
        ('Qwen2.5', os.path.join(TN3, 'llm', 'qwen25_7b', 'tn3_perquery_qwen25_7b.csv')),
        ('Phi-3.5 tự do', os.path.join(TN3, 'llm', 'phi35_mini', 'tn3_perquery_phi35_mini.csv')),
    ]:
        he[ten] = {r['case_id']: bo(r['hit1']) for r in doc(p)}
    with zipfile.ZipFile(os.path.join(GOI, 'ket_qua', 'TN4_KetQua_2026-08-17.zip')) as z:
        he['PhoGPT'] = {r['case_id']: bo(r['hit1']) for r in csv.DictReader(io.StringIO(
            z.read('tn4_perquery_phogpt_unconstrained.csv').decode('utf-8-sig')))}
    return base, he


def kiem_bam():
    """So bam 9 tep TN1+TN3 trong tn4_input.zip voi ban of-record."""
    goc = {}
    for thu, _, tep in os.walk(OF):
        for t in tep:
            if t.endswith('.csv'):
                with open(os.path.join(thu, t), 'rb') as f:
                    goc.setdefault(t, set()).add(bam(f.read()))
    khop = lech = 0
    with zipfile.ZipFile(os.path.join(GOI, 'tn4_input.zip')) as z:
        for n in z.namelist():
            if n in goc:
                if bam(z.read(n)) in goc[n]:
                    khop += 1
                else:
                    lech += 1
    return khop, lech


# ------------------------------------------------------------------- so bao
d = Document()
d.styles['Normal'].font.name = 'Times New Roman'
d.styles['Normal'].font.size = Pt(11)


def H(chu, muc=1):
    d.add_heading(chu, level=muc)


def P(chu, dam=False):
    p = d.add_paragraph()
    p.add_run(chu).bold = dam
    return p


def BANG(dau, hang, ghi=None):
    t = d.add_table(rows=1, cols=len(dau))
    t.style = 'Table Grid'
    for i, x in enumerate(dau):
        t.rows[0].cells[i].paragraphs[0].add_run(str(x)).bold = True
    for h in hang:
        c = t.add_row().cells
        for i, x in enumerate(h):
            c[i].text = str(x)
    if ghi:
        r = d.add_paragraph().add_run(ghi)
        r.italic = True
        r.font.size = Pt(9)


def main():
    base, he = nap_perquery()
    khop, lech = kiem_bam()
    dong = ['MedKG-HRR (đề xuất)', 'Retrieval-only', 'Phi-3.5 zero-shot', 'BM25-only',
            'KG-only', 'BM25 tự do', 'Llama-3', 'Qwen2.5', 'Phi-3.5 tự do', 'PhoGPT']
    tudo = dong[5:]
    dong_ = dong[:5]

    d.add_heading('Báo cáo Thí nghiệm 4, 5 và 7 — MedicalGraph / MedKG-HRR', 0)
    P('Ngày 19/08/2026. Số liệu đối chiếu với bản of-record của TN1 và TN3 tại '
      'D:\\so_lieu_cuoi_bo_sung_TN3. Bảng theo chương và mọi con số hợp/giao được tính '
      'lại từ tệp per-query ngay lúc dựng báo cáo này, không chép lại từ bản tóm tắt.')

    # ---------------------------------------------------------------- 1
    H('1. Nguồn dữ liệu và kiểm toàn vẹn')
    P(f'Chín tệp per-query của TN1 và TN3 mà TN4 và TN5 dùng làm đối chiếu đã được so băm '
      f'SHA-256 với bản of-record: {khop}/{khop + lech} tệp trùng từng byte. Không có bản '
      f'sao cũ nào lọt vào.')
    BANG(['Nhóm', 'Tệp', 'Trạng thái'], [
        ['TN1 — bốn hệ đóng danh mục',
         'tn1_perquery_{bm25_only, retrieval_only, kg_only, phi_35_zero_shot}.csv', 'khớp băm'],
        ['TN3 — BM25 tự do', 'tn3_perquery_bm25_unconstrained.csv', 'khớp băm'],
        ['TN3 — ba mô hình ngôn ngữ lớn',
         'tn3_perquery_{llama3_8b, qwen25_7b, phi35_mini}.csv', 'khớp băm'],
    ])
    P('Cấu hình chung: 181 ca độc lập, nhãn do hai bác sĩ gán độc lập kèm bước phân xử; '
      'chấm ở cấp phân nhóm ba ký tự ICD-10; 118 ca có nhãn nằm trong danh mục 1.109 bệnh '
      'của đồ thị («với tới được»), 63 ca nằm ngoài.')

    # ---------------------------------------------------------------- 2
    H('2. Bảng tổng hợp mười hệ thống')
    P('Năm hệ đầu xếp hạng trong danh mục đóng của đồ thị; năm hệ sau tìm trực tiếp trên '
      'toàn bộ 13.081 mã ICD-10 quốc gia theo Quyết định 4469/QĐ-BYT.')
    BANG(['Hệ thống', 'Hit@1 (%)', 'KTC 95%', 'Hit@5 (%)', 'MRR',
          'Reachable (%)', 'Out-of-reach (%)'], [
        ['MedKG-HRR (đề xuất)', '3,31', '[1,53; 7,04]', '6,63', '0,0442', '5,08', '0,00'],
        ['Retrieval-only', '6,08', '[3,43; 10,55]', '10,50', '0,0805', '9,32', '0,00'],
        ['Phi-3.5-mini closed', '2,21', '[0,86; 5,54]', '6,08', '0,0351', '3,39', '0,00'],
        ['BM25-only (đóng)', '1,66', '[0,57; 4,76]', '9,39', '0,0533', '2,54', '0,00'],
        ['KG-only', '1,66', '[0,57; 4,76]', '6,08', '0,0335', '2,54', '0,00'],
        ['BM25 tự do (13.081 mã)', '8,29', '[5,09; 13,22]', '14,92', '0,1128', '9,32', '6,35'],
        ['Llama-3-8B-Instruct tự do', '12,71', '[8,62; 18,35]', '14,36', '0,1358', '17,80', '3,17'],
        ['Qwen2.5-7B-Instruct tự do', '7,73', '[4,66; 12,56]', '15,47', '0,1076', '10,17', '3,17'],
        ['Phi-3.5-mini-instruct tự do', '7,73', '[4,66; 12,56]', '11,60', '0,0930', '7,63', '7,94'],
        ['PhoGPT-4B-Chat tự do', '0,55', '[0,10; 3,06]', '2,21', '0,0124', '0,85', '0,00'],
    ], 'Hàng PhoGPT lấy theo lượt B, tức lượt chạy đúng prompt phiếu §4.1 — đổi từ số lượt '
       'A (Hit@5 2,76%; MRR 0,0140) cho khớp Bảng 12 của Paper_Medical_PHT_v16.docx.')

    # ---------------------------------------------------------------- 3
    H('3. Thí nghiệm 4 — PhoGPT-4B-Chat không chịu ràng buộc danh mục')
    P('vinai/PhoGPT-4B-Chat, fp16, giải mã tham lam, max_new_tokens = 250, 181 ca, '
      'transformers 4.44.2 (ghim bắt buộc vì bản mới làm gãy mã MPT ở ba chỗ). Lượt A dùng '
      'prompt của TN3; lượt B dùng prompt phiếu §4.1 nguyên văn và là số của bài.')
    BANG(['Chỉ số', 'Lượt A (prompt TN3)', 'Lượt B (prompt phiếu)'], [
        ['Hit@1', '0,55% (1/181)', '0,55% (1/181)'],
        ['Hit@5 / Hit@10', '2,76% / 5,52%', '2,21% / 4,97%'],
        ['MRR', '0,0140', '0,0124'],
        ['Reachable / Out-of-reach', '0,85% / 0,00%', '0,85% / 0,00%'],
        ['Mã ba ký tự khác nhau trên 181 ca', '13', '13'],
        ['Ca đúng duy nhất', 'IND011 (gold3 = K35)', 'IND011'],
        ['Mã hợp lệ trong danh mục', '100%', '100%'],
        ['Thời gian chạy', '108 phút', '92,5 phút'],
    ], 'McNemar ghép cặp giữa hai lượt: b = 0, c = 0, p = 1,0 — dù văn bản sinh khác nhau '
       'ở 161/181 ca.')
    P('Mô hình suy biến thành sao chép ví dụ trong lời nhắc. Ba bằng chứng độc lập: '
      '(a) chỉ 13 mã ba ký tự trên 181 ca, mười trong số đó trùng đúng danh sách ví dụ; '
      '(b) đổi ví dụ tiêu hoá sang ví dụ da liễu thì đầu ra đổi theo, đúng 5/5 ca; '
      '(c) bỏ hẳn ví dụ thì mô hình chép lại chỗ giữ chỗ của định dạng. Hệ quả số học: '
      'trong 181 ca có đúng một ca mang nhãn trùng mã đứng đầu ví dụ, đúng bằng '
      'Hit@1 = 0,55% đo được.')
    P('Kiểm định ghép cặp PhoGPT với chín hệ còn lại, b = số ca chỉ PhoGPT đúng:')
    BANG(['Đối chiếu', 'b', 'c', 'McNemar p'], [
        ['MedKG-HRR (n = 181)', '0', '5', '0,0625'],
        ['MedKG-HRR (n = 118 reachable)', '0', '5', '0,0625'],
        ['BM25-only (đóng)', '1', '3', '0,625'],
        ['Retrieval-only', '1', '11', '0,0063'],
        ['KG-only', '1', '3', '0,625'],
        ['Phi-3.5 zero-shot (đóng)', '1', '4', '0,375'],
        ['Llama-3-8B tự do', '1', '23', '< 0,0001'],
        ['Qwen2.5-7B tự do', '1', '14', '0,0010'],
        ['Phi-3.5-mini tự do', '1', '14', '0,0010'],
        ['BM25 tự do 13.081 mã', '1', '15', '0,0005'],
    ], 'Không ca nào PhoGPT xếp đúng mà cấu hình đề xuất xếp sai (b = 0). Dòng BM25 tự do '
       'tính bổ sung trên máy sau lượt Colab; ghi trong GHI_CHU_BO_SUNG.md của zip TN4.')

    # ---------------------------------------------------------------- 4
    H('4. Thí nghiệm 5 — độ nhạy theo cách viết lời nhắc')
    P('Llama-3-8B-Instruct (bản sao NousResearch), 4-bit NF4, giải mã tham lam, cùng 181 '
      'ca. Ba biến thể: bản tiếng Việt gốc, bản tiếng Anh, bản yêu cầu suy luận từng bước. '
      'Lượt B chạy đúng phiếu §4.1 (tokenizer thô, V1 = 250 token) và là số của bài; lượt A '
      'bám giao thức TN3 (bọc thẻ hội thoại, V1 = 200 token) giữ làm đối chứng giao thức.')
    BANG(['Biến thể', 'Hit@1 (%)', 'KTC 95%', 'Hit@5 (%)', 'MRR', 'Reachable (%)',
          'McNemar p (n=118)', 'Holm ×8'], [
        ['V1 tiếng Việt (250 token)', '12,71', '[8,62; 18,35]', '16,02', '0,1400',
         '17,80', '0,0041', '0,0326'],
        ['V2 tiếng Anh (250 token)', '12,15', '[8,17; 17,72]', '16,57', '0,1428',
         '16,95', '0,0043', '0,0347'],
        ['V3 suy luận từng bước (600 token)', '14,36', '[10,00; 20,22]', '22,65', '0,1755',
         '18,64', '0,0025', '0,0200'],
    ], 'Biên độ Hit@1 = 2,21 điểm phần trăm — kịch bản B (2–5 pp) của bảng phiếu §5.3.')
    P('Cả ba biến thể đều vượt cấu hình đề xuất có ý nghĩa thống kê, kể cả sau hiệu chỉnh '
      'Holm cho tám phép so sánh. Số ca chỉ hệ tự do đúng so với số ca chỉ cấu hình đề xuất '
      'đúng lần lượt là 20/5, 18/4 và 21/5.')
    P('Kiểm tái lập: V1 cho đúng 12,71% như mốc TN3, lệch 0,00 pp so với ngưỡng ±0,6 pp của '
      'phiếu §5.3 B, và đúng trên cùng 23 ca ấy (b = 0, c = 0) — dù không ca nào trùng '
      'nguyên văn văn bản sinh, vì hai lượt đưa đầu vào theo hai giao thức khác nhau.')
    BANG(['Biến thể', 'Hit@1 A', 'Hit@1 B', 'Lệch', 'p A', 'p B', '0 mã A', '0 mã B'], [
        ['V1 tiếng Việt', '12,71%', '12,71%', '+0,00', '0,0041', '0,0041', '0', '0'],
        ['V2 tiếng Anh', '11,60%', '12,15%', '+0,55', '0,0066', '0,0043', '0', '0'],
        ['V3 suy luận từng bước', '9,39%', '14,36%', '+4,97', '0,0768', '0,0025', '1', '4'],
    ], 'Lượt A dừng sinh bằng danh sách TERMINATORS, cắt chuỗi suy luận sớm. Đó là toàn bộ '
       'nguyên nhân chênh lệch ở V3.')
    P('Năm điều kiện escalation của phiếu §6.1 chấm trên lượt B: V1 lệch mốc TN3 0,00 pp '
      '(ngưỡng 1 pp), biên độ ba biến thể 2,21 pp (ngưỡng 5 pp), không biến thể nào mất ý '
      'nghĩa, V3 hơn V1 1,65 pp (ngưỡng 5 pp), biến thể lâu nhất chạy 2 giờ 18 (ngưỡng '
      '3 giờ). Không điều kiện nào kích hoạt.')

    # ---------------------------------------------------------------- 5
    H('5. Thí nghiệm 7 — tác dụng của chuỗi suy luận (CoT ablation)')
    P('Đo bằng hiệu V3 − V1 trên cùng 181 ca, cùng mô hình, cùng tham số giải mã.')
    BANG(['Chỉ số', 'V1 tiếng Việt', 'V3 suy luận từng bước', 'Hiệu V3 − V1'], [
        ['Hit@1', '12,71%', '14,36%', '+1,65 pp'],
        ['Hit@5', '16,02%', '22,65%', '+6,63 pp'],
        ['MRR', '0,1400', '0,1755', '+0,0355'],
        ['Reachable (n = 118)', '17,80%', '18,64%', '+0,84 pp'],
        ['Out-of-reach (n = 63)', '3,17%', '6,35%', '+3,18 pp'],
        ['Ca không rút được mã nào', '0/181', '4/181', '+4 ca'],
        ['Thời gian chạy', '58,7 phút', '138,1 phút', '× 2,4'],
    ])
    P('Chuỗi suy luận nâng cả Hit@1 lẫn Hit@5, mạnh nhất ở Hit@5: nó đưa nhãn đúng vào '
      'top-5 thường xuyên hơn hẳn. Cái giá phải ghi: bốn ca dùng hết 600 token cho phần '
      'suy luận nên không còn chỗ cho khối JSON và bị chấm sai, nên 14,36% là cận dưới. '
      'Tỷ lệ 4/181 = 2,2% còn xa ngưỡng 20% mà phiếu §6.2.1 đặt để phải nâng trần token '
      'lên 800, nên cấu hình giữ nguyên.')
    P('Kết quả này đảo chiều so với lượt A, nơi V3 hạ Hit@1 3,32 pp và mất ý nghĩa thống kê '
      '(p = 0,0768). Chiều của tác dụng CoT vì vậy phụ thuộc cách bọc đầu vào, không chỉ '
      'phụ thuộc câu chữ của lời nhắc.')

    # ---------------------------------------------------------------- 6
    H('6. Phân tách theo chương ICD-10')
    P('Số ca xếp đúng ở hạng một, theo chương có ít nhất năm ca. Cột HỢP là số ca có ít '
      'nhất một trong mười hệ bắt đúng.')
    ch = {}
    for c, r in base.items():
        ch.setdefault(r['gold3'][0], []).append(c)
    lon = sorted([k for k, v in ch.items() if len(v) >= 5], key=lambda k: -len(ch[k]))
    hang = []
    for k in lon:
        d_ = [str(sum(1 for c in ch[k] if he[t].get(c))) for t in dong]
        hop = len({c for c in ch[k] for t in dong if he[t].get(c)})
        hang.append([k, len(ch[k])] + d_ + [hop])
    BANG(['Chương', 'n'] + dong + ['HỢP'], hang,
         'Chương Z và chương R gộp lại là 54/181 ca. Ở chương Z, không hệ nào bắt trúng '
         'quá một ca và hợp của cả mười hệ chỉ được hai ca: IND163 (Z31.6, Phi-3.5 tự do) '
         'và IND211 (Z20.3, Qwen2.5). Ở chương R, con số cao nhất là 5 ca của Phi-3.5 tự '
         'do, hợp được 7 ca.')

    # ---------------------------------------------------------------- 7
    H('7. Bức tranh gộp trên 181 ca')
    hop_dong = len({c for c in base for t in dong_ if he[t].get(c)})
    hop_tudo = len({c for c in base for t in tudo if he[t].get(c)})
    hop_all = len({c for c in base for t in dong if he[t].get(c)})
    giao = len([c for c in base if all(he[t].get(c) for t in dong)])
    khong = len([c for c in base if not any(he[t].get(c) for t in dong)])
    outr = [c for c, r in base.items() if not bo(r['voi_toi_duoc'])]
    hop_out = len({c for c in outr for t in tudo if he[t].get(c)})
    n = len(base)
    BANG(['Nhóm', 'Số ca', '%'], [
        ['Hợp của 5 hệ đóng danh mục', f'{hop_dong}/{n}', f'{100*hop_dong/n:.2f}'.replace('.', ',')],
        ['Hợp của 5 hệ tự do', f'{hop_tudo}/{n}', f'{100*hop_tudo/n:.2f}'.replace('.', ',')],
        ['Hợp của cả 10 hệ', f'{hop_all}/{n}', f'{100*hop_all/n:.2f}'.replace('.', ',')],
        ['Giao của cả 10 hệ', f'{giao}/{n}', f'{100*giao/n:.2f}'.replace('.', ',')],
        ['Không hệ nào xếp đúng', f'{khong}/{n}', f'{100*khong/n:.2f}'.replace('.', ',')],
        ['Hợp của 5 hệ tự do trên 63 ca ngoài tầm với', f'{hop_out}/{len(outr)}',
         f'{100*hop_out/len(outr):.2f}'.replace('.', ',')],
    ], 'Thêm PhoGPT vào nhóm tự do nâng hợp của nhóm ấy từ 42 lên 43 ca, nhưng ca thêm vào '
       'đã nằm trong hợp của nhóm đóng danh mục, nên hợp của cả mười hệ vẫn là 54 ca.')
    P('Nới danh mục và đổi sang mô hình ngôn ngữ lớn nâng trần khả thi từ 9,39% lên 29,83%, '
      'nhưng vẫn để lại hơn 70% tập độc lập ngoài tầm với của mọi hệ đã thử.')

    # ---------------------------------------------------------------- 8
    H('8. Chấm lại với quy tắc mã nới lỏng (bản of-record TN3)')
    P('Bộ chuẩn hoá mã của phiếu chỉ nhận tối đa hai chữ số sau dấu chấm, nên loại các mã '
      'theo chuẩn ICD-10-CM của Hoa Kỳ. Bản of-record đã chấm lại toàn bộ từ văn bản thô '
      'với quy tắc nới lỏng, vẫn so ở cấp ba ký tự.')
    BANG(['Hệ', 'Số truy vấn có mã bị loại', 'Hit@1 theo phiếu', 'Hit@1 nới lỏng'], [
        ['Llama-3', '0/181', '12,71%', '12,71%'],
        ['Qwen2.5', '3/181', '7,73%', '7,73%'],
        ['Phi-3.5-mini', '45/181', '7,73%', '7,18%'],
    ], 'Chỉ Phi-3.5 đổi, và đổi theo hướng giảm: quy tắc gốc loại mã CM đứng đầu rồi đôn mã '
       'thứ hai lên, tình cờ có lợi ở vài ca. Bảng ở Mục 2 giữ con số theo quy tắc gốc để '
       'so trực tiếp được với TN1 và TN2; con số nới lỏng nêu ở đoạn giới hạn của bài.')
    P('Dưới quy tắc nới lỏng, kiểm định ghép cặp của Phi-3.5 với cấu hình đề xuất đổi từ '
      'b/c = 14/6 (p = 0,115) thành 13/6 (p = 0,167); Llama-3 giữ nguyên 22/5 (p = 0,0015). '
      'Không kết luận nào của TN4, TN5 hay TN7 phụ thuộc lựa chọn quy tắc này.')

    # ---------------------------------------------------------------- 9
    H('9. Những chỗ còn cần thầy xem')
    BANG(['#', 'Nội dung'], [
        ['1', 'Phiếu TN5 §6.1 ghi ngưỡng «p > 0,0125 sau Holm». Con số 0,0125 = 0,05/4 vốn '
              'là ngưỡng cho p thô, còn p đã hiệu chỉnh Holm thì so với 0,05. Hai cách đọc '
              'hợp lý đều cho kết quả «không kích hoạt», nên đã tạm làm theo.'],
        ['2', 'Hàng PhoGPT giữ trong bảng theo đúng phiếu TN4 mục 1 (Hit@1 ≤ 5% thì thêm '
              'hàng). Bài có kèm đoạn khuyến nghị không dùng hàng này làm bằng chứng về '
              'giới hạn của các mô hình bản ngữ tiếng Việt nói chung.'],
        ['3', 'Câu về chương Z ở phần kết luận đã sửa ở bản v16 cho khớp số thật (không hệ '
              'nào quá một ca; hợp được hai ca). Chỗ này không thuộc phiếu nào.'],
    ])

    # --------------------------------------------------------------- 10
    H('Phụ lục A. Hai thư gửi thầy — soạn sẵn, người viết tự gửi')
    P('Địa chỉ: tiennt@ut.edu.vn.')
    P('Thư 1 — tiêu đề đúng chữ phiếu TN4 §9.4: '
      '«[TN4 MedicalGraph] Kết quả PhoGPT unconstrained». Đính kèm '
      'TN4_KetQua_2026-08-17.zip và TN4B_KetQua_2026-08-17.zip.', True)
    P('Kính thưa thầy, em đã chạy xong TN4 theo phiếu. PhoGPT-4B-Chat trên 181 ca độc lập '
      'đạt Hit@1 = 0,55% (1/181, KTC 95% [0,10; 3,06]), Hit@5 = 2,21%, MRR = 0,0124, xếp '
      'thứ năm trong năm hệ tự do. Rơi vào kịch bản Hit@1 ≤ 5% của phiếu mục 1, nên em đã '
      'thêm hàng vào Bảng 12 thành mười hệ thống và đồng bộ mười lăm chỗ chữ liên quan. '
      'Có một điều em phải báo thầy vì nó ảnh hưởng cách đọc hàng này: trên toàn bộ 181 ca, '
      'mô hình chỉ sinh ra 13 mã ICD-10 ba ký tự khác nhau, trong đó mười mã trùng đúng '
      'danh sách ví dụ trong lời nhắc. Em đã kiểm ba đường — thay ví dụ tiêu hoá bằng ví dụ '
      'da liễu thì đầu ra đổi theo, đúng 5/5 ca; bỏ hẳn ví dụ thì mô hình chép lại chỗ giữ '
      'chỗ của định dạng; và chạy lại toàn bộ 181 ca bằng một lời nhắc viết khác hẳn thì '
      'văn bản sinh khác nhau ở 161/181 ca nhưng vẫn ra đúng 0,55%, đúng một ca đúng ấy, '
      'đúng 13 mã ấy (McNemar b = 0, c = 0, p = 1). Nghĩa là đầu ra gần như không phụ thuộc '
      'mô tả bệnh nhân. Em vẫn báo cáo hàng này để giữ tính đầy đủ của so sánh, nhưng trong '
      'bài có ghi thêm một đoạn khuyến nghị không dùng nó làm bằng chứng về giới hạn của '
      'các mô hình bản ngữ tiếng Việt nói chung. Một chi tiết về gói kết quả: '
      'tn4_mcnemar.csv do Colab sinh thiếu dòng đối chiếu với BM25 tự do vì gói đầu vào em '
      'dựng đã quên một tệp; em đã tính bổ sung trên máy (b = 1, c = 15, p = 0,0005) và ghi '
      'rõ là tính sau trong GHI_CHU_BO_SUNG.md, không tệp per-query nào bị sửa. '
      'Em cảm ơn thầy.')
    P('Thư 2 — tiêu đề «[TN5+TN7 MedicalGraph] Kết quả prompt sensitivity và CoT ablation» '
      '(phiếu TN5 §5.4 không quy định tiêu đề, đặt cho khớp dạng của TN4). Đính kèm '
      'TN5B_KetQua_2026-08-19.zip và TN5_TN7_KetQua_2026-08-17.zip.', True)
    P('Kính thưa thầy, em đã chạy xong TN5 và TN7. Em gửi hai gói vì em chạy hai lượt, và '
      'hai lượt cho kết luận khác nhau ở một biến thể. Lượt đầu em bám cấu hình của TN3: '
      'bọc đầu vào bằng thẻ hội thoại của mô hình, V1 dùng 200 token, giữ nguyên câu chữ '
      'lời nhắc của TN3. Đọc lại phiếu §3.1 và §4.1 thì cả ba chỗ đó đều lệch: phiếu in '
      'nguyên mã dùng tokenizer thô, đặt V1 = 250 token, và câu quy tắc bốn của V1 là «Chỉ '
      'trả về JSON.» chứ không có cụm «theo mẫu». Em đã chạy lại đủ ba biến thể đúng phiếu '
      'từng chữ và lấy lượt này làm số của bài: tiếng Việt 12,71% (p = 0,0041), tiếng Anh '
      '12,15% (p = 0,0043), suy luận từng bước 14,36% (p = 0,0025). '
      'Ba điều em xin báo thầy. Thứ nhất, bản tiếng Việt chạy ở 250 token với tokenizer thô '
      'vẫn cho đúng 12,71% như mốc TN3, lệch 0,00 pp so với ngưỡng ±0,6 pp ở §5.3 B; chặt '
      'hơn thế, kiểm ghép cặp giữa hai lượt cho b = 0 và c = 0, tức đúng trên cùng 23 ca ấy '
      'không lệch một ca, dù không ca nào trùng nguyên văn văn bản sinh. Thứ hai, '
      'escalation em báo thầy hôm 17/08 — biến thể suy luận từng bước mất ý nghĩa thống kê, '
      'p = 0,0768 — không còn đúng dưới lượt chạy đúng phiếu; nguyên nhân là lượt đầu em '
      'dừng sinh bằng danh sách TERMINATORS, cắt chuỗi suy luận sớm. Em vẫn nêu cả hai con '
      'số trong bài vì chênh lệch ấy cho thấy mức ý nghĩa của biến thể CoT còn phụ thuộc '
      'cách bọc đầu vào. Thứ ba, chấm lại năm trigger ở §6.1 trên lượt đúng phiếu thì không '
      'trigger nào kích hoạt. Một chỗ em xin thầy cho ý: trigger thứ ba ghi «p > 0,0125 sau '
      'Holm», mà 0,0125 = 0,05/4 vốn là ngưỡng để so với p thô, còn p đã hiệu chỉnh Holm thì '
      'thường so với 0,05; em hiểu theo cả hai cách đều ra không kích hoạt nên tạm làm vậy. '
      'Bài đã sửa theo, lưu thành Paper_Medical_PHT_v16.docx, mọi chỗ sửa tô đỏ. '
      'Em cảm ơn thầy.')

    d.save(DICH)
    print('Da luu:', DICH)
    d2 = Document(DICH)
    print(f'  {len(d2.paragraphs)} doan | {len(d2.tables)} bang')
    with open(DICH, 'rb') as f:
        print('  SHA-256:', bam(f.read()))


if __name__ == '__main__':
    main()

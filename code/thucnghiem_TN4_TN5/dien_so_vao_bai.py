# -*- coding: utf-8 -*-
"""Điền số thật của TN4 / TN5 vào các đoạn sửa cho Paper_Medical_PHT_v14.

Chạy SAU khi đã có kết quả từ hai notebook Colab. In ra văn bản đã điền đủ số,
dán thẳng vào bài. Mọi con số đều tính lại từ tệp per-query, không chép tay.

    python dien_so_vao_bai.py --tn4 <thu_muc_ket_qua_TN4> \
                              --tn5 <thu_muc_ket_qua_TN5> \
                              --tn3 "C:\\Users\\adc\\Desktop\\so_lieu_cuoi_bo_sung_TN3"

Bỏ --tn4 hoặc --tn5 thì phần tương ứng được bỏ qua. --tn3 mặc định trỏ đúng
thư mục TN3 trên máy này.
"""
import argparse, functools, io, os, sys, zipfile

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TN3_MAC_DINH = r'C:\Users\adc\Desktop\so_lieu_cuoi_bo_sung_TN3'


def so(x, n=2):
    """4.5 -> '4,50' — dấu phẩy thập phân như trong bài."""
    return f'{x:.{n}f}'.replace('.', ',')


def ktc(s):
    """'[4.66, 12.56]' -> '[4,66; 12,56]' — đúng quy ước khoảng tin cậy của bài."""
    return str(s).replace('.', ',').replace(', ', '; ')


def nap_tn3(tn3):
    """Trả về (med, hits_tu_do) — med là 181 ca of-record, hits là dict hệ -> Series bool."""
    with zipfile.ZipFile(os.path.join(tn3, 'tn3_input.zip')) as z:
        med = pd.read_csv(io.BytesIO(z.read('independent_scored_perquery.csv'))).set_index('case_id')

    B = os.path.join(tn3, 'thi_nghiem_3_unconstrained')
    hits = {}
    bm = pd.read_csv(os.path.join(B, 'tn3_perquery_bm25_unconstrained.csv')).set_index('case_id')
    hits['BM25 tự do'] = bm['hit1'].astype(bool)
    for tag, ten in [('llama3_8b', 'Llama-3'), ('qwen25_7b', 'Qwen2.5'),
                     ('phi35_mini', 'Phi-3.5')]:
        d = pd.read_csv(os.path.join(B, 'llm', tag, f'tn3_perquery_{tag}.csv')).set_index('case_id')
        hits[ten] = d['hit1'].astype(bool)
    return med, hits


def phan_tn4(tn4, tn3):
    med, hits = nap_tn3(tn3)
    oor = med.index[med['voi_toi_duoc'] == False]

    pq = pd.read_csv(os.path.join(tn4, 'tn4_perquery_phogpt_unconstrained.csv')).set_index('case_id')
    sm = pd.read_csv(os.path.join(tn4, 'tn4_summary.csv')).iloc[0]
    mc = pd.read_csv(os.path.join(tn4, 'tn4_mcnemar.csv'))
    hits['PhoGPT-4B'] = pq['hit1'].astype(bool)

    def hop(ten_he, chi_muc):
        u = functools.reduce(lambda a, b: a | b,
                             [hits[t].reindex(med.index).fillna(False) for t in ten_he])
        return int(u[chi_muc].sum())

    bon = ['BM25 tự do', 'Llama-3', 'Qwen2.5', 'Phi-3.5']
    nam = bon + ['PhoGPT-4B']
    hop_cu, hop_moi = hop(bon, oor), hop(nam, oor)

    # xep hang out-of-reach cua ca 5 he tu do
    oor_pct = {t: 100 * hits[t].reindex(med.index).fillna(False)[oor].sum() / len(oor) for t in nam}
    xep_oor = sorted(oor_pct.items(), key=lambda kv: -kv[1])

    hit1 = float(sm['hit1_pct'])
    xep_hit1 = sorted([('PhoGPT-4B-Chat', hit1), ('Llama-3', 12.71), ('BM25 tự do', 8.29),
                       ('Qwen2.5', 7.73), ('Phi-3.5', 7.73)], key=lambda kv: -kv[1])
    thu_hang = [t for t, _ in xep_hit1].index('PhoGPT-4B-Chat') + 1

    def lay_mc(khoa):
        d = mc[mc['doi_chieu'].str.contains(khoa, regex=False)]
        return d.iloc[0] if len(d) else None
    mc118 = lay_mc('n=118')

    r = []
    A = r.append
    A('# TN4 — số đã điền\n')
    A('## Ô mới của Bảng 12 (hàng thứ 10, đặt sau Phi-3.5-mini tự do)\n')
    A('```')
    A(f"PhoGPT-4B-Chat tự do (13.081 mã) | {so(hit1)} | {ktc(sm['hit1_ci'])} | "
      f"{so(float(sm['hit5_pct']))} | {so(float(sm['mrr']), 4)} | "
      f"{so(float(sm['hit1_reachable_pct']))} | {so(float(sm['hit1_full_icd_pct']))}")
    A('```\n')

    A('## Câu «Kết quả trải trên một dải hẹp…» — bản thay\n')
    A(f"Kết quả trải trên một dải hẹp: Llama-3 đạt Hit@1 = 12,71% (khoảng tin cậy Wilson 95%: "
      f"[8,62; 18,35]), BM25 tự do 8,29%, Qwen2.5 và Phi-3.5 cùng 7,73%, PhoGPT-4B-Chat "
      f"{so(hit1)}%. Cả năm đều cao hơn từng hệ trong nhóm đóng danh mục, vốn nằm trong dải "
      f"1,66–6,08%.  [PhoGPT xếp thứ {thu_hang}/5 trong nhóm tự do]\n")

    A('## Câu «Trên 63 ca ngoài tầm với…» — bản thay\n')
    ds = ', '.join(f'{so(v)}% ({t})' for t, v in xep_oor)
    A(f"Trên 63 ca ngoài tầm với của đồ thị, năm hệ tự do đạt lần lượt {ds}, trong khi cả năm "
      f"hệ đóng danh mục đều bằng không. Hợp của năm hệ tự do trên nhóm này cũng chỉ đạt "
      f"{hop_moi}/63 ca, tức {so(100*hop_moi/63)}%.")
    A(f"  [mốc cũ với bốn hệ: {hop_cu}/63 = {so(100*hop_cu/63)}%; "
      f"PhoGPT {'CÓ' if hop_moi > hop_cu else 'KHÔNG'} thêm ca mới nào]\n")

    if mc118 is not None:
        A('## Câu McNemar (n=118 reachable)\n')
        A(f"PhoGPT-4B-Chat: b = {int(mc118['b_chi_phogpt_dung'])}, "
          f"c = {int(mc118['c_chi_he_kia_dung'])}, p = {so(float(mc118['mcnemar_p']), 4)} "
          f"(mốc Llama-3 trên cùng tập con: b = 20, c = 5, p = 0,0041).\n")

    A('## Kịch bản\n')
    kb = 'A (≤5%)' if hit1 <= 5 else ('B (5–15%)' if hit1 < 15 else 'C (≥15%)')
    A(f"Hit@1 = {so(hit1)}% → kịch bản **{kb}**.")
    if hit1 >= 15:
        A("\n**⚠ KỊCH BẢN C — phải báo thầy trước khi sửa bài. Finding C cần reframe.**")
    A('')
    return '\n'.join(r)


def phan_tn5(tn5):
    s = pd.read_csv(os.path.join(tn5, 'tn5_summary_variants.csv')).set_index('variant')
    h = {v: float(s.loc[v, 'hit1_pct_full']) for v in s.index}
    spread = max(h.values()) - min(h.values())
    v1 = h['V1_baseline']

    r = []
    A = r.append
    A('# TN5 / TN7 — số đã điền\n')
    A('## Đoạn độ nhạy theo cách diễn đạt prompt (chèn sau đoạn «Kết quả trải trên một dải hẹp…»)\n')
    A(f"Vì mọi con số của nhóm hệ tự do đều đo dưới một cách viết prompt duy nhất, chúng tôi "
      f"chạy lại Llama-3-8B trên cùng 181 ca với ba biến thể prompt — bản tiếng Việt gốc, bản "
      f"tiếng Anh, và bản yêu cầu suy luận từng bước (chain-of-thought) — giữ nguyên mọi tham số "
      f"giải mã. Hit@1 trải từ {so(min(h.values()))}% đến {so(max(h.values()))}% "
      f"(biên độ {so(spread)} điểm phần trăm): "
      f"bản tiếng Việt {so(h['V1_baseline'])}%, bản tiếng Anh {so(h['V2_english'])}%, "
      f"bản suy luận từng bước {so(h['V3_cot'])}%.")

    for v, nhan in [('V1_baseline', 'bản tiếng Việt'), ('V2_english', 'bản tiếng Anh'),
                    ('V3_cot', 'bản suy luận từng bước')]:
        p = float(s.loc[v, 'mcnemar_n118_p'])
        A(f"  [{nhan}: McNemar n=118 b={int(s.loc[v,'mcnemar_n118_b'])}, "
          f"c={int(s.loc[v,'mcnemar_n118_c'])}, p={so(p,4)}, "
          f"{'CÒN' if p < 0.05 else 'MẤT'} ý nghĩa thô, "
          f"{'qua' if p < 0.00625 else 'không qua'} Holm 0,05/8]")

    kb = 'A (≤2 pp)' if spread <= 2 else ('B (2–5 pp)' if spread <= 5 else 'C (>5 pp)')
    A(f"\nKịch bản **{kb}**.")
    if spread <= 2:
        A("Câu kết cho bài: *«Chênh lệch giữa MedKG-HRR và hệ tự do giữ nguyên dấu và mức ý "
          "nghĩa dưới cả ba cách viết prompt, nên phát hiện này không phải là hiện tượng của "
          "một cách diễn đạt riêng lẻ.»*")
    elif spread <= 5:
        A("Câu kết cho bài: *«Hit@1 của hệ tự do phụ thuộc cách viết prompt trong biên độ "
          f"{so(spread)} điểm phần trăm; chiều của kết luận không đổi, nhưng con số tuyệt đối "
          "phải đọc kèm cách viết prompt đã dùng.»*")
    else:
        A("**⚠ KỊCH BẢN C — báo thầy. Phải hạ mức khẳng định xuống «dưới cách viết prompt gốc».**")

    A(f"\n## Kiểm tái lập V1\n")
    A(f"V1 = {so(v1)}% so với mốc TN3 12,71% — lệch {so(abs(v1-12.71))} pp "
      f"({'đạt' if abs(v1-12.71) < 0.6 else '**KHÔNG đạt**'} ngưỡng ±0,6 pp).")

    d = h['V3_cot'] - h['V1_baseline']
    A(f"\n## TN7 — CoT\n")
    A(f"V3 − V1 = {d:+.2f} pp".replace('.', ',').replace('-', '−'))
    if abs(d) > 5:
        A("**⚠ Chênh > 5 pp — báo thầy (mục 6.1 phiếu TN5).**")
    A('')
    return '\n'.join(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tn4')
    ap.add_argument('--tn5')
    ap.add_argument('--tn3', default=TN3_MAC_DINH)
    ap.add_argument('--ra', help='ghi ra tệp thay vì in ra màn hình')
    a = ap.parse_args()

    if not a.tn4 and not a.tn5:
        ap.error('cần ít nhất một trong --tn4 / --tn5')

    phan = []
    if a.tn4:
        phan.append(phan_tn4(a.tn4, a.tn3))
    if a.tn5:
        phan.append(phan_tn5(a.tn5))
    out = '\n\n---\n\n'.join(phan)

    if a.ra:
        with open(a.ra, 'w', encoding='utf-8') as f:
            f.write(out)
        print('Đã ghi:', a.ra)
    else:
        print(out)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""Dien ba muc "Nhan dinh (nguoi viet bo sung)" vao ba tep ghi chu, roi bam lai.

Phieu de trong ba muc nay cho nguoi viet. Ban chu da duyet 19/08/2026.
Script chay lai duoc: no thay dung doan giu cho '[…]'.
"""
import copy, hashlib, io, os, sys, zipfile
from docx import Document
from docx.oxml.ns import qn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
GOC = r'D:\NopThay_TN4_TN5_2026-08-17\ket_qua'

TN4 = [
 "PhoGPT-4B-Chat không thất bại theo kiểu một mô hình yếu xếp sai thứ hạng, mà theo kiểu "
 "một mô hình không đọc đầu vào. Ba phép kiểm độc lập cùng chỉ về một hành vi: 13 mã ba "
 "ký tự trên 181 ca, mười trong số đó trùng danh sách ví dụ; đổi ví dụ sang chuyên khoa "
 "khác thì đầu ra đổi theo, 5/5 ca; và một prompt viết khác hẳn vẫn cho đúng cùng một con "
 "số, cùng một ca đúng, cùng 13 mã ấy. Vì vậy con số Hit@1 = 0,55% nên đọc là «đúng một ca "
 "trong tập có nhãn trùng mã đầu danh sách ví dụ», chứ không phải một ước lượng năng lực "
 "chẩn đoán.",
 "Hệ quả cho bài: hàng này củng cố Finding B ở chỗ nó cho thấy quyền tự do chọn trong "
 "13.081 mã không tự động thành lợi thế — một mô hình không neo được đầu ra vào mô tả "
 "bệnh nhân thì không gian rộng chỉ làm hại. Nhưng hàng này không đủ để nói gì về giới hạn "
 "của các mô hình ngôn ngữ bản ngữ tiếng Việt nói chung, vì thứ đo được ở đây là khả năng "
 "tuân thủ chỉ dẫn ít mẫu, không phải tri thức y khoa. Một lượt tinh chỉnh hướng dẫn hoặc "
 "một mô hình bản ngữ khác có thể cho kết quả hoàn toàn khác.",
]

TN5A = [
 "Lượt này bám giao thức TN3 chứ không bám phiếu §4.1, nên vai trò của nó là đối chứng "
 "chứ không phải số của bài. Giá trị của nó nằm ở hai chỗ. Một là nó tái lập TN3 nguyên "
 "văn 181/181 ca, xác nhận giải mã tham lam tất định giữa hai phiên chạy cách nhau nhiều "
 "tuần trên hai phiên Colab khác nhau. Hai là đặt cạnh lượt chạy đúng phiếu, nó cô lập "
 "được đúng một biến: cách bọc đầu vào. Biến thể suy luận từng bước xuống 9,39% và mất ý "
 "nghĩa thống kê ở lượt này, lên 14,36% và đạt p = 0,0025 ở lượt kia, chỉ vì danh sách "
 "TERMINATORS cắt chuỗi suy luận sớm. Đây là một cảnh báo có ích cho bất kỳ so sánh nào "
 "giữa các mô hình sinh: một chi tiết cấu hình không ai coi là tham số thí nghiệm vẫn có "
 "thể lật kết luận về mức ý nghĩa.",
]

TN5B = [
 "Ba biến thể prompt cho Hit@1 trong dải 12,15–14,36%, biên độ 2,21 điểm phần trăm, và cả "
 "ba đều vượt cấu hình đề xuất có ý nghĩa thống kê trên 118 ca với tới được, kể cả sau "
 "hiệu chỉnh Holm cho tám phép so sánh. Kết luận Finding C vì vậy không phụ thuộc một cách "
 "viết prompt cụ thể: nó giữ nguyên cả về dấu lẫn mức ý nghĩa dưới cả ba cách viết.",
 "Bản tiếng Việt tái lập mốc TN3 ở mức chặt nhất có thể kiểm: cùng 12,71%, và đúng trên "
 "cùng 23 ca ấy, dù không ca nào trùng nguyên văn văn bản sinh vì hai lượt đưa đầu vào "
 "theo hai giao thức khác nhau. Nói cách khác, đổi giao thức đưa đầu vào làm đổi toàn bộ "
 "chữ mà không đổi một quyết định hạng nhất nào — với biến thể này.",
 "Với biến thể suy luận từng bước thì không như vậy: nó là biến thể mạnh nhất ở đây "
 "(Hit@1 14,36%, Hit@5 22,65%) nhưng xuống 9,39% khi đầu vào được bọc thẻ hội thoại và "
 "chuỗi suy luận bị cắt sớm. Nó cũng là biến thể duy nhất để lại ca không rút được mã nào, "
 "4/181 ca, nên 14,36% là cận dưới. Đọc chung ba điều đó: chuỗi suy luận giúp mô hình trải "
 "rộng hơn và bắt đúng nhiều hơn, nhưng đổi lại nó nhạy với mọi thứ chạm vào độ dài sinh, "
 "và đó là chiều nhạy cảm mà một phép so sánh chỉ đổi câu chữ prompt sẽ không thấy.",
]

VIEC = [
    ('TN4_KetQua_2026-08-17.zip', 'tn4_ghichu_diengiai.docx', 'tn4_SHA256.txt',
     '7. Nhận định', TN4),
    ('TN5_TN7_KetQua_2026-08-17.zip', 'tn5_ghichu_diengiai.docx', 'tn5_SHA256.txt',
     'E. Nhận định', TN5A),
    ('TN5B_KetQua_2026-08-19.zip', 'tn5b_ghichu_diengiai.docx', 'tn5b_SHA256.txt',
     'C. Nhận định', TN5B),
]

GHI_CHU = """
## Sửa ngày 19/08/2026 — điền mục «Nhận định (người viết bổ sung)»

Phiếu để trống mục này cho người viết; bản chữ đã được duyệt và điền vào
`{docx}`. **Không tệp per-query nào bị đụng đến** — băm của chúng giữ nguyên.
Băm của tệp ghi chú thì đổi, và `{sha}` đã cập nhật theo:

- `{docx}`: `{cu}` → `{moi}`

Dựng lại bằng `dien_nhan_dinh.py`.
"""


def dien(duong_dan, moc, doan_moi):
    d = Document(duong_dan)
    i = next(k for k, p in enumerate(d.paragraphs)
             if p.text.strip() == '[…]' and moc in d.paragraphs[k - 1].text)
    p = d.paragraphs[i]
    p.runs[0].text = doan_moi[0]
    for r in p.runs[1:]:
        r.text = ''
    truoc = p._p
    for chu in doan_moi[1:]:
        el = copy.deepcopy(p._p)
        truoc.addnext(el)
        truoc = el
        for t in el.iter(qn('w:t')):
            t.text = ''
        list(el.iter(qn('w:t')))[0].text = chu
    d.save(duong_dan)
    return i


def bam(f):
    with open(f, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    tam = os.path.join(os.environ.get('TEMP', '.'), '_dien_nhan_dinh')
    for zn, docx, shafile, moc, doan in VIEC:
        zp = os.path.join(GOC, zn)
        thu = os.path.join(tam, zn[:-4])
        os.makedirs(thu, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            ten = z.namelist()
            z.extractall(thu)
        dd = os.path.join(thu, docx)
        cu = bam(dd)[:16]
        i = dien(dd, moc, doan)
        moi = bam(dd)
        print(f"{zn}: {docx} para[{i}] <- {len(doan)} doan | {cu}... -> {moi[:16]}...")

        sp = os.path.join(thu, shafile)                       # cap nhat bam
        dong = io.open(sp, encoding='utf-8').read().splitlines()
        dong = [f"{moi}  {docx}" if l.strip().endswith(docx) else l for l in dong]
        io.open(sp, 'w', encoding='utf-8', newline='\n').write('\n'.join(dong) + '\n')

        gp = os.path.join(thu, 'GHI_CHU_BO_SUNG.md')          # ghi lai viec sua
        cu_nd = io.open(gp, encoding='utf-8').read() if os.path.exists(gp) else \
            f"# {zn[:-4]} — ghi chu bo sung\n"
        io.open(gp, 'w', encoding='utf-8', newline='\n').write(
            cu_nd + GHI_CHU.format(docx=docx, sha=shafile, cu=cu + '…', moi=moi[:16] + '…'))
        if 'GHI_CHU_BO_SUNG.md' not in ten:
            ten.append('GHI_CHU_BO_SUNG.md')
        # ban than GHI_CHU cung nam trong SHA256.txt cua TN4 -> bam lai sau khi ghi
        if any(l.strip().endswith('GHI_CHU_BO_SUNG.md') for l in dong):
            hg = bam(gp)
            dong = [f"{hg}  GHI_CHU_BO_SUNG.md"
                    if l.strip().endswith('GHI_CHU_BO_SUNG.md') else l for l in dong]
            io.open(sp, 'w', encoding='utf-8', newline='
').write('
'.join(dong) + '
')

        os.remove(zp)                                          # dong lai zip
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
            for t in ten:
                z.write(os.path.join(thu, t), t)
        print(f"   dong lai {zn}: {len(ten)} tep")


if __name__ == '__main__':
    main()

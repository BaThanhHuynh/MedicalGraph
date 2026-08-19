# -*- coding: utf-8 -*-
"""Dong goi tn4_input.zip / tn5_input.zip + MANIFEST cho phien TN4-TN5."""
import os, zipfile, hashlib, shutil, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r'C:\Users\adc\Desktop\so_lieu_cuoi_bo_sung_TN3'
OUT  = r'D:\NopThay_TN4_TN5_2026-08-17'
TMP  = os.path.join(OUT, '_tmp_input')
os.makedirs(OUT, exist_ok=True)
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)

# 6 tep goc cua tn3_input.zip
with zipfile.ZipFile(os.path.join(BASE, 'tn3_input.zip')) as z:
    z.extractall(TMP)

# them 3 tep per-query cua TN3 => cho phep kiem tai lap V1 tung ca + McNemar cheo
for tag in ['llama3_8b', 'qwen25_7b', 'phi35_mini']:
    shutil.copy(os.path.join(BASE, 'thi_nghiem_3_unconstrained', 'llm', tag,
                             f'tn3_perquery_{tag}.csv'), TMP)

# BM25 tu do: phieu TN4 muc 8 doi McNemar voi DU BON he tu do. Thieu tep nay thi
# cell McNemar bo qua BM25 ma khong bao gi - da dinh mot lan roi, dung bo lai.
shutil.copy(os.path.join(BASE, 'thi_nghiem_3_unconstrained',
                         'tn3_perquery_bm25_unconstrained.csv'), TMP)

ten_tep = sorted(os.listdir(TMP))
for zname in ['tn4_input.zip', 'tn5_input.zip']:
    zp = os.path.join(OUT, zname)
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in ten_tep:
            z.write(os.path.join(TMP, f), f)
    print(f'{zname}: {len(ten_tep)} tep, {os.path.getsize(zp)/1e6:.2f} MB')

# MANIFEST
lines = ['# MANIFEST SHA-256 — goi TN4 + TN5/TN7, dung ngay 2026-08-17', '',
         '## Noi dung goi input (tn4_input.zip == tn5_input.zip)', '']
for f in ten_tep:
    with open(os.path.join(TMP, f), 'rb') as fh:
        lines.append(f'{hashlib.sha256(fh.read()).hexdigest()}  {f}')
lines += ['', '## Notebook', '']
for f in sorted(os.listdir(OUT)):
    if f.endswith(('.ipynb', '.zip', '.md', '.py')):
        with open(os.path.join(OUT, f), 'rb') as fh:
            lines.append(f'{hashlib.sha256(fh.read()).hexdigest()}  {f}')

with open(os.path.join(OUT, 'MANIFEST_SHA256.txt'), 'w', encoding='utf-8', newline='\n') as fh:
    fh.write('\n'.join(lines) + '\n')

shutil.rmtree(TMP)
print('\n'.join(lines))

"""Aday köşeleri görselleştirir (Stage 2 geliştirme aracı).

Kullanım:
    python view_candidates.py dev_10
    python view_candidates.py data/personal/s332501004_07.jpg

Çıktı: results/<ad>_candidates.png
    kırmızı nokta = aday (find_candidates çıktısı)
    yeşil daire   = elle işaretlenmiş doğru köşe
Bu araç köşe SEÇMEZ; sadece adayları gösterir.
"""
import json
import os
import sys

import cv2
import numpy as np

from pipeline.stage1_preprocess import preprocess
from pipeline.stage2_corners import find_candidates, harris_response

name = sys.argv[1]
if name.endswith(".jpg"):
    img_path, json_path = name, name[:-4] + ".json"
    name = os.path.basename(name)[:-4]
else:
    img_path = f"data/dev/images/{name}.jpg"
    json_path = f"data/dev/annotations/{name}.json"

img = cv2.imread(img_path)
if img is None:
    sys.exit(f"Fotoğraf okunamadı: {img_path}")

norm, scale = preprocess(img)
cands = find_candidates(harris_response(norm))
print(f"{name}: aday sayısı = {len(cands)}")

out = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
for x, y, _ in cands:
    cv2.circle(out, (x, y), 3, (0, 0, 255), -1)

if os.path.exists(json_path) and cands:
    pts = np.array([(x, y) for x, y, _ in cands], dtype=float)
    with open(json_path) as f:
        gt = json.load(f)["corners"]
    for label, (gx, gy) in zip(["TL", "TR", "BR", "BL"], gt):
        g = np.array([(gx + 0.5) * scale - 0.5, (gy + 0.5) * scale - 0.5])
        d = np.linalg.norm(pts - g, axis=1).min()
        cv2.circle(out, (int(round(g[0])), int(round(g[1]))), 14, (0, 200, 0), 2)
        print(f"   {label}: en yakın aday {d:5.1f} px uzakta")

os.makedirs("results", exist_ok=True)
cv2.imwrite(f"results/{name}_candidates.png", out)
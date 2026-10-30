"""Stage 3'ü doğru (elle işaretlenmiş) köşelerle dener.

Kullanım:
    python test_stage3.py dev_07                              # deneme setinden
    python test_stage3.py data/personal/s332501004_09.jpg     # kendi fotoğrafından

Çıktı: results/<ad>_stage3.png  (A3 | A4 | H4 yan yana, üzerinde ideal hücre çizgileri)
Not: Geliştirme aracıdır; teslim edilen programda köşeler Harris'ten gelecek.
"""
import json
import os
import sys

import cv2

from pipeline.stage3_rectify import CELL, SIZE, rectify

name = sys.argv[1]
if name.endswith(".jpg"):
    img_path, json_path = name, name[:-4] + ".json"
    name = os.path.basename(name)[:-4]
else:
    img_path = f"data/dev/images/{name}.jpg"
    json_path = f"data/dev/annotations/{name}.json"

with open(json_path) as f:
    corners = json.load(f)["corners"]
img = cv2.imread(img_path)
if img is None:
    sys.exit(f"Fotoğraf okunamadı: {img_path}")

r = rectify(img, corners)
print(f"{name}:  A3 4. köşe hatası = {r['a3_corner_err_px']:7.2f} px   "
      f"A4 RMS = {r['a4_rms_px']:6.2f} px   H4 köşe hatası = {r['h4_max_err_px']:.1e} px")

# Üç görüntüyü yan yana koy; ideal hücre sınırlarını (her CELL pikselde) yeşil çiz
panels = []
for key in ("rect_A3", "rect_A4", "rect_H4"):
    p = r[key].copy()
    for k in range(10):
        c = int(round(k * CELL - 0.5)) if k else 0
        c = min(c, SIZE - 1)
        cv2.line(p, (c, 0), (c, SIZE - 1), (0, 200, 0), 1)
        cv2.line(p, (0, c), (SIZE - 1, c), (0, 200, 0), 1)
    cv2.putText(p, key[-2:], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    panels.append(p)
gap = 255 * (panels[0][:, :10] * 0 + 1)
out = cv2.hconcat([panels[0], gap, panels[1], gap, panels[2]])
os.makedirs("results", exist_ok=True)
cv2.imwrite(f"results/{name}_stage3.png", out)
"""Harris'i keşfetme aracı: Stage 1 + cv2.cornerHarris çalıştırır, sonucu görselleştirir.

Kullanım:
    python explore_harris.py dev_07
    python explore_harris.py data/personal/s332501004_09.jpg
    python explore_harris.py dev_07 --block 5 --ksize 3 --k 0.04 --thr 0.01

Çıktı: results/<ad>_harris_explore.png  (3 panel)
    1. Ön işlenmiş görüntü
    2. Harris yanıt haritası (parlak = güçlü köşe)
    3. Eşiği geçen pikseller (kırmızı) + elle işaretlenmiş doğru köşeler (yeşil daire)

Not: Bu araç köşe SEÇMEZ; sadece Harris'in ne gördüğünü gösterir.
Aday çıkarma (yerel maksimum bastırma) ve 4 köşenin seçimi Stage 2'de yazılacak.
"""
import argparse
import json
import os

import cv2
import numpy as np

from pipeline.stage1_preprocess import preprocess

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--block", type=int, default=5, help="blockSize: komşuluk penceresi")
ap.add_argument("--ksize", type=int, default=3, help="Sobel çekirdeği (3, 5 veya 7)")
ap.add_argument("--k", type=float, default=0.04, help="Harris k sabiti")
ap.add_argument("--thr", type=float, default=0.01, help="eşik: maksimum yanıtın bu oranı")
a = ap.parse_args()

if a.name.endswith(".jpg"):
    img_path, json_path = a.name, a.name[:-4] + ".json"
    name = os.path.basename(a.name)[:-4]
else:
    img_path = f"data/dev/images/{a.name}.jpg"
    json_path = f"data/dev/annotations/{a.name}.json"
    name = a.name

img = cv2.imread(img_path)
if img is None:
    raise SystemExit(f"Fotoğraf okunamadı: {img_path}")

norm, scale = preprocess(img)

# Harris: girdi float32 olmalı. Mürekkep koyu, kâğıt açık; Harris için fark etmez (gradyana bakar).
R = cv2.cornerHarris(np.float32(norm), a.block, a.ksize, a.k)

above = R > a.thr * R.max()
print(f"{name}: çalışma boyutu {norm.shape[1]}x{norm.shape[0]}, "
      f"eşiği geçen piksel sayısı = {int(above.sum())}")

# Panel 2: yanıt haritası. Sadece pozitif kısmı (köşeler), karekök ile sıkıştırılmış ki zayıflar da görünsün
Rpos = np.sqrt(np.clip(R, 0, None))
heat = cv2.applyColorMap(np.uint8(255 * Rpos / (Rpos.max() + 1e-12)), cv2.COLORMAP_INFERNO)

# Panel 3: eşiği geçen pikseller kırmızı, doğru köşeler yeşil
overlay = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
overlay[above] = (0, 0, 255)
if os.path.exists(json_path):
    with open(json_path) as f:
        for x, y in json.load(f)["corners"]:
            p = (int(round((x + 0.5) * scale - 0.5)), int(round((y + 0.5) * scale - 0.5)))
            cv2.circle(overlay, p, 12, (0, 200, 0), 2)

gap = np.full((norm.shape[0], 10, 3), 255, np.uint8)
out = cv2.hconcat([cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR), gap, heat, gap, overlay])
os.makedirs("results", exist_ok=True)
cv2.imwrite(f"results/{name}_harris_explore.png", out)
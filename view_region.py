"""find_puzzle_region sonucunu görselleştirir (Stage 2 geliştirme aracı).

Kullanım:
    python view_region.py dev_10
    python view_region.py data/personal/s332501004_07.jpg
    python view_region.py all          # deneme setinin 20 fotoğrafı

Çıktı: results/<ad>_region.png  (mavi = bulunan bölge, yeşil daire = doğru köşeler)
Bu araç sadece gösterir; bölge bulma ve köşe seçimi stage2_corners.py'de.
"""
import json
import os
import sys

import cv2
import numpy as np

from pipeline.stage1_preprocess import preprocess
from pipeline.stage2_corners import find_puzzle_region


def run(arg):
    if arg.endswith(".jpg"):
        img_path, json_path = arg, arg[:-4] + ".json"
        name = os.path.basename(arg)[:-4]
    else:
        img_path = f"data/dev/images/{arg}.jpg"
        json_path = f"data/dev/annotations/{arg}.json"
        name = arg
    img = cv2.imread(img_path)
    if img is None:
        print(f"{name}: fotoğraf okunamadı")
        return
    norm, scale = preprocess(img)
    quad = find_puzzle_region(norm)
    out = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    msg = "bölge BULUNAMADI" if quad is None else "bölge bulundu (karşılaştıracak JSON yok)"
    if quad is not None:
        cv2.polylines(out, [np.int32(quad).reshape(-1, 1, 2)], True, (255, 0, 0), 3)
    if os.path.exists(json_path):
        with open(json_path) as f:
            gt = (np.array(json.load(f)["corners"]) + 0.5) * scale - 0.5
        for gx, gy in gt:
            cv2.circle(out, (int(round(gx)), int(round(gy))), 14, (0, 200, 0), 2)
        if quad is not None:
            d = max(np.linalg.norm(quad - g, axis=1).min() for g in gt)
            msg = f"bölge köşelerinin doğru köşelere en büyük uzaklığı = {d:6.1f} px"
    print(f"{name}: {msg}")
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(f"results/{name}_region.png", out)


if sys.argv[1] == "all":
    for i in range(1, 21):
        run(f"dev_{i:02d}")
else:
    run(sys.argv[1])
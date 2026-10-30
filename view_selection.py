"""select_corners sonucunu doğru köşelerle karşılaştırır (Stage 2 geliştirme aracı).

Kullanım:
    python view_selection.py dev_10
    python view_selection.py data/personal/s332501004_09.jpg
    python view_selection.py all        # deneme seti (20) + kişisel fotoğraflar

Hata, ORİJİNAL fotoğraf piksellerinde ölçülür.
Sıralama henüz yoksa her doğru köşeye en yakın seçilen köşe kullanılır.
Çıktı: results/<ad>_selection.png  (kırmızı = seçilen, yeşil = doğru)
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

from pipeline.stage1_preprocess import preprocess, to_original
from pipeline.stage2_corners import (find_candidates, find_puzzle_region,
                                     harris_response, select_corners)


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
        print(f"{name:15s} fotoğraf okunamadı")
        return None
    norm, scale = preprocess(img)
    region = find_puzzle_region(norm)
    cands = find_candidates(harris_response(norm))
    corners = select_corners(region, cands, norm.shape[1])
    if corners is None:
        print(f"{name:15s} köşe SEÇİLEMEDİ")
        return None
    pts = to_original(corners, scale)
    out = img.copy()
    r = max(4, img.shape[1] // 150)
    for x, y in pts:
        cv2.circle(out, (int(round(x)), int(round(y))), r, (0, 0, 255), -1)
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(json_path):
        print(f"{name:15s} seçildi (karşılaştıracak JSON yok)")
        cv2.imwrite(f"results/{name}_selection.png", out)
        return None
    with open(json_path) as f:
        gt = np.array(json.load(f)["corners"])
    errs = [np.linalg.norm(pts - g, axis=1).min() for g in gt]
    for x, y in gt:
        cv2.circle(out, (int(round(x)), int(round(y))), 3 * r, (0, 200, 0), max(2, r // 2))
    cv2.imwrite(f"results/{name}_selection.png", out)
    print(f"{name:15s} köşe hataları (px): " + "  ".join(f"{e:6.1f}" for e in errs)
          + f"   en büyük = {max(errs):6.1f}")
    return max(errs)


if sys.argv[1] == "all":
    items = [f"dev_{i:02d}" for i in range(1, 21)] + sorted(glob.glob("data/personal/*.jpg"))
    worst = [w for w in (run(a) for a in items) if w is not None]
    if worst:
        print(f"\nBaşarılı: {len(worst)}/{len(items)}   en büyük hataların ortancası = {np.median(worst):.1f} px")
else:
    run(sys.argv[1])
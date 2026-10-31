"""Stage 5 ve 6'yı elle işaretlenmiş köşelerle dener (köşe hatası karışmasın diye).

Kullanım:
    python test_stage5.py dev                 # deneme seti
    python test_stage5.py personal            # kişisel dataset
    python test_stage5.py dev --raw           # hücre hazırlığı OLMADAN (rapor için 'önce' durumu)

Ölçülenler (ödevin Bölüm 7 tanımları):
    hücre doğruluğu: 81 hücreden (rakam ya da boş) doğru tahmin edilenlerin oranı, fotoğraflar üzerinden ortalama
    boş-hücre F1: dolu/boş kararının F1 skoru (dolu = pozitif sınıf)
    tam eşleşme: 81 hücresi de doğru olan fotoğrafların oranı
    karışıklıklar: en sık yanlış okunan (doğru → okunan) çiftler
Geliştirme aracıdır; teslim edilen programda köşeler Harris'ten gelecek.
"""
import collections
import glob
import json
import sys
import time

import cv2

from pipeline.stage3_rectify import rectify
from pipeline.stage5_cells_ocr import recognize_grid
from pipeline.stage6_validate import validate

which = sys.argv[1]
prepare = "--raw" not in sys.argv
if which == "dev":
    items = [(f"data/dev/images/dev_{i:02d}.jpg", f"data/dev/annotations/dev_{i:02d}.json") for i in range(1, 21)]
else:
    items = [(p, p[:-4] + ".json") for p in sorted(glob.glob("data/personal/*.jpg"))]

accs, exact = [], 0
tp = fp = fn = 0
confusions = collections.Counter()
for img_path, json_path in items:
    with open(json_path) as f:
        ann = json.load(f)
    img = cv2.imread(img_path)
    t0 = time.time()
    rect = rectify(img, ann["corners"])["rect_H4"]
    grid, filled = recognize_grid(rect, prepare=prepare)
    dt = time.time() - t0
    gt = ann["grid"]
    correct = 0
    for r in range(9):
        for c in range(9):
            g, p = gt[r][c], grid[r][c]
            correct += g == p
            if g and p != g:
                confusions[(g, p)] += 1
            if not g and p:
                confusions[(0, p)] += 1
            is_f, gt_f = filled[r][c], g != 0
            tp += is_f and gt_f
            fp += is_f and not gt_f
            fn += (not is_f) and gt_f
    acc = correct / 81
    accs.append(acc)
    exact += correct == 81
    verdict, conf = validate(grid)
    name = img_path.split("/")[-1][:-4]
    print(f"{name:15s} hücre doğruluğu %{100 * acc:5.1f}   karar: {verdict:7s}  süre {dt:4.1f} s")

prec = tp / (tp + fp) if tp + fp else 0
rec = tp / (tp + fn) if tp + fn else 0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
print(f"\n{'HAZIRLIKLI' if prepare else 'HAZIRLIKSIZ'}:  ortalama hücre doğruluğu %{100 * sum(accs) / len(accs):.1f}"
      f"   boş-hücre F1 = {f1:.3f}   tam eşleşme = {exact}/{len(items)}")
print("En sık karışıklıklar (doğru → okunan, 0 = boş):",
      ", ".join(f"{g}→{p} ×{n}" for (g, p), n in confusions.most_common(8)) or "yok")
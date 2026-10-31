"""Stage 5: hücre kenar payının (MARGIN_FRAC) etkisi.

sudoku_scan.py çıktısındaki H4 görüntülerini farklı kenar paylarıyla yeniden okur ve
etiketle karşılaştırır. Diğer bütün Stage 5 ayarları sabit kalır.

Kullanım:
    python margin_study.py --results results/personal --labels data/personal
    python margin_study.py --results results/dev --labels data/dev/annotations
"""
import argparse
import glob
import json
import os

import cv2

import pipeline.stage5_cells_ocr as s5

MARGINS = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24]
CHOSEN = 0.12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()

    photos = []
    for ann_path in sorted(glob.glob(os.path.join(args.labels, "*.json"))):
        if ann_path.endswith("_result.json"):
            continue
        ann = json.load(open(ann_path))
        name = os.path.splitext(ann["image"])[0]
        rect = cv2.imread(os.path.join(args.results, f"{name}_rect_H4.png"))
        if rect is not None:
            photos.append((rect, ann["grid"]))
    print(f"{len(photos)} fotoğraf\n")
    print(f"{'pay':>6s} {'pay(px)':>8s} {'hücre doğr. %':>14s} {'boş-F1':>7s} {'boş→dolu':>9s} {'dolu→boş':>9s} {'yanlış rakam':>13s}")

    for m in MARGINS:
        s5.MARGIN_FRAC = m
        correct = total = tp = fp = fn = wrong_digit = 0
        for rect, gt in photos:
            grid, _ = s5.recognize_grid(rect)
            for r in range(9):
                for c in range(9):
                    g, p = gt[r][c], grid[r][c]
                    total += 1
                    correct += g == p
                    tp += bool(g) and bool(p)
                    fp += (not g) and bool(p)
                    fn += bool(g) and (not p)
                    wrong_digit += bool(g) and bool(p) and g != p
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
        px = int(round(m * s5.CELL))
        mark = "  ← seçilen" if m == CHOSEN else ""
        print(f"{m:>6.2f} {px:>8d} {100 * correct / total:>14.2f} {f1:>7.3f} {fp:>9d} {fn:>9d} {wrong_digit:>13d}{mark}")
    s5.MARGIN_FRAC = CHOSEN


if __name__ == "__main__":
    main()
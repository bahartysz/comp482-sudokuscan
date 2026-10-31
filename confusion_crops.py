"""Bölüm 5 ve 7 için: yanlış okunan hücrelerin örnek kesitleri.

sudoku_scan.py çıktısındaki H4 görüntüsünü (<ad>_rect_H4.png) ve <ad>_grid.txt
dosyasını etiketle karşılaştırır, yanlış okunan her hücreyi kesip tek bir şekilde toplar.

Kullanım:
    python confusion_crops.py --results results/personal --labels data/personal
    python confusion_crops.py --results results/dev --labels data/dev/annotations --out rapor_gorselleri/karisiklik_dev.png

Çıktı:
    terminalde yanlış hücrelerin listesi ve karışıklık sayıları
    rapor_gorselleri/karisiklik_ornekleri.png  (her kesitin başlığı: fotoğraf, doğru → okunan)
"""
import argparse
import collections
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt

CELL = 50
PAD = 5          # kesite komşu hücreden biraz pay ekle ki bağlam görünsün


def label_text(v):
    return "boş" if v == 0 else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="sudoku_scan.py çıktı klasörü")
    ap.add_argument("--labels", required=True, help="etiket JSON klasörü")
    ap.add_argument("--out", default="rapor_gorselleri/karisiklik_ornekleri.png")
    ap.add_argument("--max", type=int, default=24, help="şekilde en fazla kaç kesit")
    args = ap.parse_args()

    wrong = []
    for ann_path in sorted(glob.glob(os.path.join(args.labels, "*.json"))):
        if ann_path.endswith("_result.json"):
            continue
        ann = json.load(open(ann_path))
        name = os.path.splitext(ann["image"])[0]
        grid_path = os.path.join(args.results, f"{name}_grid.txt")
        rect_path = os.path.join(args.results, f"{name}_rect_H4.png")
        if not (os.path.exists(grid_path) and os.path.exists(rect_path)):
            print(name, ": sonuç yok (not_found?)")
            continue
        pred = [[int(ch) for ch in line.strip()] for line in open(grid_path) if line.strip()]
        rect = cv2.cvtColor(cv2.imread(rect_path), cv2.COLOR_BGR2RGB)
        h, w = rect.shape[:2]
        for r in range(9):
            for c in range(9):
                g, p = ann["grid"][r][c], pred[r][c]
                if g != p:
                    y0, x0 = max(0, r * CELL - PAD), max(0, c * CELL - PAD)
                    y1, x1 = min(h, (r + 1) * CELL + PAD), min(w, (c + 1) * CELL + PAD)
                    wrong.append((name, r, c, g, p, rect[y0:y1, x0:x1]))

    counts = collections.Counter((g, p) for _, _, _, g, p, _ in wrong)
    print(f"\nToplam yanlış hücre: {len(wrong)}")
    print("Karışıklıklar (doğru → okunan):",
          ", ".join(f"{label_text(g)}→{label_text(p)} ×{n}" for (g, p), n in counts.most_common()))
    for name, r, c, g, p, _ in wrong:
        print(f"  {name}  satır {r + 1} sütun {c + 1}:  {label_text(g)} → {label_text(p)}")

    if not wrong:
        print("Yanlış hücre yok, şekil çizilmedi.")
        return

    shown = wrong[:args.max]
    cols = min(8, len(shown))
    rows = (len(shown) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, r, c, g, p, crop) in zip(axes.ravel(), shown):
        ax.imshow(crop)
        short = name.split("_")[-1]
        ax.set_title(f"_{short} ({r + 1},{c + 1})\n{label_text(g)} → {label_text(p)}", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print("\nKaydedildi:", args.out)


if __name__ == "__main__":
    main()
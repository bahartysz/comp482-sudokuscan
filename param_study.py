"""Stage 2 parametre çalışması: blockSize, ksize, k ve THR'nin
aday sayısına ve köşe hatasına etkisi.

Kontrollü deney: her seferinde TEK parametre değişir, diğerleri
seçilen değerlerde (DEFAULTS) sabit kalır.

Kullanım:
    python param_study.py --images data/personal --labels data/personal
    python param_study.py --images data/dev/images --labels data/dev/annotations

Çıktılar:
    terminalde her parametre değeri için özet tablo
    rapor_gorselleri/param_study.csv          (fotoğraf bazında bütün sonuçlar)
    rapor_gorselleri/param_study.png          (4 parametre için grafik)
"""
import argparse
import csv
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import pipeline.stage2_corners as s2
from pipeline.stage1_preprocess import preprocess, to_original

# Seçilen (şu an kullanılan) değerler. Bir parametre taranırken diğerleri bunlarda kalır.
DEFAULTS = {"HARRIS_BLOCK": 5, "HARRIS_KSIZE": 3, "HARRIS_K": 0.04, "THR": 0.005}

# Denenecek değerler
SWEEPS = {
    "HARRIS_BLOCK": [3, 5, 7, 9],
    "HARRIS_KSIZE": [3, 5, 7],
    "HARRIS_K": [0.02, 0.04, 0.06, 0.10],
    "THR": [0.001, 0.002, 0.005, 0.01, 0.02],
}


def corner_error(pred, gt):
    """Ödev Bölüm 7: 4 köşenin ortalama uzaklığı / gerçek kenar uzunluğu (%)."""
    gt = np.array(gt, dtype=float)
    side = np.mean([np.linalg.norm(gt[k] - gt[(k + 1) % 4]) for k in range(4)])
    return 100 * np.mean(np.linalg.norm(np.array(pred, dtype=float) - gt, axis=1)) / side


def run_once(norm, scale, gt):
    """Stage 2'yi o anki parametrelerle çalıştırır → (aday sayısı, köşe hatası % ya da None)."""
    R = s2.harris_response(norm)
    candidates = s2.find_candidates(R)
    region = s2.find_puzzle_region(norm)
    corners = s2.order_corners(s2.select_corners(region, candidates, norm.shape[1]))
    if corners is None:
        return len(candidates), None
    return len(candidates), corner_error(to_original(corners, scale), gt)


def set_params(params):
    for key, value in params.items():
        setattr(s2, key, value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()

    # Fotoğrafları ve etiketleri eşleştir; ön işlemeyi bir kez yap (parametrelerden bağımsız)
    photos = []
    for ann_path in sorted(glob.glob(os.path.join(args.labels, "*.json"))):
        if ann_path.endswith("_result.json"):
            continue
        ann = json.load(open(ann_path))
        img = cv2.imread(os.path.join(args.images, ann["image"]))
        if img is None:
            print("okunamadı:", ann["image"])
            continue
        norm, scale = preprocess(img)
        photos.append((os.path.splitext(ann["image"])[0], norm, scale, ann["corners"]))
    print(f"{len(photos)} fotoğraf\n")

    rows = []
    for param, values in SWEEPS.items():
        print(f"=== {param}  (diğerleri: {', '.join(f'{k}={v}' for k, v in DEFAULTS.items() if k != param)})")
        print(f"{'değer':>8s} {'ort. aday':>10s} {'ort. hata %':>12s} {'en kötü %':>10s} {'bulunan':>8s}")
        for value in values:
            set_params({**DEFAULTS, param: value})
            counts, errors, found = [], [], 0
            for name, norm, scale, gt in photos:
                n, err = run_once(norm, scale, gt)
                counts.append(n)
                if err is not None:
                    errors.append(err)
                    found += 1
                rows.append({"param": param, "value": value, "image": name,
                             "candidates": n, "corner_err_pct": "" if err is None else round(err, 3)})
            mean_e = f"{np.mean(errors):.2f}" if errors else "-"
            max_e = f"{np.max(errors):.2f}" if errors else "-"
            mark = "  ← seçilen" if value == DEFAULTS[param] else ""
            print(f"{value:>8} {np.mean(counts):>10.1f} {mean_e:>12s} {max_e:>10s} {found:>4d}/{len(photos)}{mark}")
        print()
    set_params(DEFAULTS)

    os.makedirs("rapor_gorselleri", exist_ok=True)
    with open("rapor_gorselleri/param_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["param", "value", "image", "candidates", "corner_err_pct"])
        w.writeheader()
        w.writerows(rows)

    # Grafik: her parametre için ortalama köşe hatası (sol eksen) ve ortalama aday sayısı (sağ eksen)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (param, values) in zip(axes, SWEEPS.items()):
        errs, cnts = [], []
        for v in values:
            sel = [r for r in rows if r["param"] == param and r["value"] == v]
            e = [r["corner_err_pct"] for r in sel if r["corner_err_pct"] != ""]
            errs.append(np.mean(e) if e else np.nan)
            cnts.append(np.mean([r["candidates"] for r in sel]))
        x = range(len(values))
        ax.plot(x, errs, "o-", color="tab:blue", label="köşe hatası (%)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(param)
        ax.set_ylabel("ort. köşe hatası (%)", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(x, cnts, "s--", color="tab:orange", label="aday sayısı")
        ax2.set_ylabel("ort. aday sayısı", color="tab:orange")
        ax.axvline(values.index(DEFAULTS[param]), color="gray", linestyle=":")
        ax.set_title(param)
    plt.tight_layout()
    plt.savefig("rapor_gorselleri/param_study.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Kaydedildi: rapor_gorselleri/param_study.csv, rapor_gorselleri/param_study.png")


if __name__ == "__main__":
    main()
"""Ödevin Bölüm 7 metriklerini hesaplar: sudoku_scan.py çıktılarını etiketlerle karşılaştırır.

Kullanım:
    python sudoku_scan.py --batch data/dev/images --outdir results/dev
    python evaluate.py --results results/dev --labels data/dev/annotations

    python sudoku_scan.py --batch data/personal --outdir results/personal
    python evaluate.py --results results/personal --labels data/personal

Metrikler (Bölüm 7):
    köşe hatası      : 4 köşenin ortalama Öklid uzaklığı / gerçek kenar uzunluğu (%)
    hücre doğruluğu  : 81 hücreden (rakam ya da boş) doğru olanların oranı, fotoğraflar üzerinden ortalama
    boş-hücre F1     : dolu/boş kararının F1 skoru (dolu = pozitif)
    tam eşleşme      : 81 hücresi de doğru olan fotoğrafların oranı
    sağlamlık        : çökmeden ve 20 s içinde işlenen fotoğrafların oranı
Hepsi toplamda ve C1–C5 koşullarına göre ayrı ayrı yazdırılır (rapor Bölüm 6 için).
"""
import argparse
import glob
import json
import os

import numpy as np

TIME_LIMIT = 20.0


def image_metrics(res, ann):
    """Tek fotoğrafın metrikleri. Bulunamadıysa köşe hatası None, grid tamamen boş sayılır."""
    gt_grid = ann["grid"]
    gt_c = np.array(ann["corners"], dtype=float)
    side = np.mean([np.linalg.norm(gt_c[k] - gt_c[(k + 1) % 4]) for k in range(4)])
    found = res is not None and res.get("verdict") != "not_found" and res.get("corners")
    corner = None
    if found:
        pc = np.array(res["corners"], dtype=float)
        corner = 100 * np.mean(np.linalg.norm(pc - gt_c, axis=1)) / side
    grid = res["grid"] if found else [[0] * 9 for _ in range(9)]
    correct = tp = fp = fn = 0
    for r in range(9):
        for c in range(9):
            g, p = gt_grid[r][c], grid[r][c]
            correct += g == p
            tp += bool(g) and bool(p)
            fp += (not g) and bool(p)
            fn += bool(g) and (not p)
    robust = res is not None and res.get("runtime_s", 1e9) <= TIME_LIMIT
    return {"corner": corner, "acc": correct / 81, "tp": tp, "fp": fp, "fn": fn,
            "exact": correct == 81, "robust": robust, "found": bool(found)}


def summarize(rows):
    if not rows:
        return "-"
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    corners = [r["corner"] for r in rows if r["corner"] is not None]
    ce = f"%{np.mean(corners):.2f}" if corners else "-"
    return (f"n={len(rows):2d}  köşe hatası {ce:>6s}  hücre doğruluğu %{100 * np.mean([r['acc'] for r in rows]):5.1f}"
            f"  boş-F1 {f1:.3f}  tam eşleşme {sum(r['exact'] for r in rows)}/{len(rows)}"
            f"  bulunan {sum(r['found'] for r in rows)}/{len(rows)}  sağlamlık {sum(r['robust'] for r in rows)}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="sudoku_scan.py çıktı klasörü")
    ap.add_argument("--labels", required=True, help="etiket (JSON) klasörü")
    args = ap.parse_args()

    rows, by_cond = [], {}
    for ann_path in sorted(glob.glob(os.path.join(args.labels, "*.json"))):
        if ann_path.endswith("_result.json"):
            continue
        with open(ann_path) as f:
            ann = json.load(f)
        name = os.path.splitext(ann["image"])[0]
        res_path = os.path.join(args.results, f"{name}_result.json")
        res = json.load(open(res_path)) if os.path.exists(res_path) else None
        m = image_metrics(res, ann)
        rows.append(m)
        for cond in ann.get("conditions", []):
            by_cond.setdefault(cond, []).append(m)
        ce = f"%{m['corner']:.2f}" if m["corner"] is not None else "bulunamadı"
        print(f"{name:15s} {'+'.join(ann.get('conditions', [])):6s} köşe {ce:>10s}  hücre %{100 * m['acc']:5.1f}"
              f"{'  TAM' if m['exact'] else ''}")
    print("\nTOPLAM   " + summarize(rows))
    for cond in sorted(by_cond):
        print(f"{cond:8s} " + summarize(by_cond[cond]))


if __name__ == "__main__":
    main()
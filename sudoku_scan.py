"""SudokuScan: bir sudoku fotoğrafından köşeleri, düzeltilmiş görüntüleri ve 9x9 grid'i çıkarır.

Kullanım (ödev Bölüm 6):
    python sudoku_scan.py --input photo.jpg --outdir out/
    python sudoku_scan.py --batch images_dir/ --outdir results/

Her <ad> fotoğrafı için çıktı klasörüne yazılanlar:
    <ad>_grid.txt        9 satır, 9 rakam, ayraç yok, boş hücre = 0 (H4 rektifikasyonundan)
    <ad>_result.json     köşeler, A3/A4/H4 matrisleri, A3 4. köşe hatası, A4 RMS, grid, karar, süre
    <ad>_harris.png      Harris yanıt haritası; bütün adaylar ve seçilen 4 köşe işaretli
    <ad>_rect_A3.png, <ad>_rect_A4.png, <ad>_rect_H4.png   450x450 düzeltilmiş görüntüler
    <ad>_debug.png       H4 görüntüsü + hücre sınırları + okunan rakamlar + çakışan hücreler
Bulmaca bulunamazsa ya da herhangi bir hata olursa program çökmez; "verdict": "not_found" yazar.
"""
import argparse
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np

from pipeline.stage1_preprocess import preprocess, to_original
from pipeline.stage2_corners import (find_candidates, find_puzzle_region,
                                     harris_response, order_corners,
                                     select_corners)
from pipeline.stage3_rectify import CELL, SIZE, rectify
from pipeline.stage5_cells_ocr import recognize_grid
from pipeline.stage6_validate import validate

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def harris_debug(norm, R, candidates, corners):
    """Harris yanıt haritasını renklendirir; adaylar küçük camgöbeği, seçilen köşeler büyük kırmızı."""
    Rpos = np.sqrt(np.clip(R, 0, None))                       # zayıf yanıtlar da görünsün diye karekök
    heat = cv2.applyColorMap(np.uint8(255 * Rpos / (Rpos.max() + 1e-12)), cv2.COLORMAP_INFERNO)
    out = cv2.addWeighted(cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR), 0.35, heat, 0.65, 0)
    for x, y, _ in candidates:
        cv2.circle(out, (int(x), int(y)), 2, (255, 255, 0), -1)
    if corners is not None:
        for label, (x, y) in zip(["TL", "TR", "BR", "BL"], corners):
            p = (int(round(x)), int(round(y)))
            cv2.circle(out, p, 9, (0, 0, 255), 2)
            cv2.putText(out, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def grid_debug(rect_h4, grid, conflicts):
    """H4 görüntüsü üzerine hücre sınırları, okunan rakamlar (mavi) ve çakışan hücreler (kırmızı)."""
    out = rect_h4.copy()
    for (r, c) in conflicts:
        overlay = out.copy()
        cv2.rectangle(overlay, (c * CELL, r * CELL), ((c + 1) * CELL - 1, (r + 1) * CELL - 1), (0, 0, 255), -1)
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for k in range(10):
        t = 2 if k % 3 == 0 else 1
        p = min(k * CELL, SIZE - 1)
        cv2.line(out, (p, 0), (p, SIZE - 1), (0, 160, 0), t)
        cv2.line(out, (0, p), (SIZE - 1, p), (0, 160, 0), t)
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                cv2.putText(out, str(grid[r][c]), (c * CELL + CELL - 16, r * CELL + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 60, 0), 2)
    return out


def not_found(name, t0, reason):
    """Bulmaca bulunamadığında yazılacak sonuç."""
    return {"image": name, "conditions": [], "corners": None, "A3": None, "A4": None, "H4": None,
            "a3_corner_err_px": None, "a4_rms_px": None, "grid": None,
            "verdict": "not_found", "reason": reason, "runtime_s": round(time.time() - t0, 3)}


def process_image(img_path, outdir):
    """Tek bir fotoğrafı işler ve bütün çıktı dosyalarını yazar. Asla istisna fırlatmaz."""
    t0 = time.time()
    file_name = os.path.basename(img_path)
    name = os.path.splitext(file_name)[0]
    result_path = os.path.join(outdir, f"{name}_result.json")
    try:
        img = cv2.imread(img_path)
        if img is None:
            result = not_found(file_name, t0, "image could not be read")
        else:
            # Stage 1 + Stage 2 (küçültülmüş çalışma görüntüsünde)
            norm, scale = preprocess(img)
            R = harris_response(norm)
            candidates = find_candidates(R)
            region = find_puzzle_region(norm)
            corners_work = order_corners(select_corners(region, candidates, norm.shape[1]))
            cv2.imwrite(os.path.join(outdir, f"{name}_harris.png"),
                        harris_debug(norm, R, candidates, corners_work))
            if corners_work is None:
                result = not_found(file_name, t0, "puzzle corners not found")
            else:
                corners = to_original(corners_work, scale)          # orijinal fotoğraf pikselleri
                # Stage 3
                r = rectify(img, corners)
                for key in ("A3", "A4", "H4"):
                    cv2.imwrite(os.path.join(outdir, f"{name}_rect_{key}.png"), r[f"rect_{key}"])
                # Stage 5 + 6
                grid, _ = recognize_grid(r["rect_H4"])
                verdict, conflicts = validate(grid)
                with open(os.path.join(outdir, f"{name}_grid.txt"), "w") as f:
                    f.write("\n".join("".join(str(v) for v in row) for row in grid) + "\n")
                cv2.imwrite(os.path.join(outdir, f"{name}_debug.png"), grid_debug(r["rect_H4"], grid, conflicts))
                result = {
                    "image": file_name,
                    "conditions": [],
                    "corners": [[float(x), float(y)] for x, y in corners],
                    "A3": r["A3"].tolist(),
                    "A4": r["A4"].tolist(),
                    "H4": r["H4"].tolist(),
                    "a3_corner_err_px": round(r["a3_corner_err_px"], 3),
                    "a4_rms_px": round(r["a4_rms_px"], 3),
                    "grid": grid,
                    "verdict": verdict,
                    "conflicts": [list(rc) for rc in conflicts],
                    "runtime_s": round(time.time() - t0, 3),
                }
    except Exception as e:                       # beklenmeyen her hata: çökme yerine not_found
        traceback.print_exc()
        result = not_found(file_name, t0, f"error: {type(e).__name__}: {e}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description="SudokuScan")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="tek bir fotoğraf")
    src.add_argument("--batch", help="fotoğrafların bulunduğu klasör")
    ap.add_argument("--outdir", required=True, help="çıktı klasörü")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.input:
        paths = [args.input]
    else:
        paths = sorted(os.path.join(args.batch, f) for f in os.listdir(args.batch) if f.endswith(IMAGE_EXTS))
    for p in paths:
        res = process_image(p, args.outdir)
        print(f"{os.path.basename(p)}: {res['verdict']}  ({res['runtime_s']:.1f} s)")


if __name__ == "__main__":
    sys.exit(main())
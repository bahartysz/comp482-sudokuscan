"""Etiketleme aracı: kişisel fotoğraflar için ground-truth JSON üretir (Appendix A formatı).

Kullanım:
    python annotate.py data/personal/s332501004_01.jpg

Akış:
    1. Genel görünümde köşelere sırayla tıkla: TL, TR, BR, BL (kalın dış çerçevenin DIŞ köşesi).
    2. Her tıklamadan sonra büyütülmüş pencere açılır; tam köşeye bir kez daha tıkla.
    3. Dört köşe bitince önizleme gelir: 'y' kaydet, 'r' baştan başla, ESC çık.
    4. Terminalde bulmacayı (P1/P2/P3) ve koşulları (ör. C2 C5) gir.
    5. Fotoğrafın yanına aynı adla .json dosyası yazılır.

Not: Bu araç teslim edilen pipeline'ın parçası değildir; sadece etiket hazırlamak içindir.
"""
import json
import os
import sys

import cv2

# Basılı bulmacaların verilen rakamları (0 = boş hücre). Kâğıtlardan okunup sudoku kurallarıyla doğrulandı.
PUZZLES = {
    "P1": ["401839000", "830002000", "900000001", "000140273", "060070908",
           "010008504", "080007000", "045080700", "000600032"],   # daily-sudoku.com, 28.10.2026 Medium
    "P2": ["006002304", "904750082", "008006005", "003000040", "200400830",
           "407500000", "000600008", "700020453", "000370069"],   # sudoku-online.org #1001
    "P3": ["001607800", "000491500", "500203040", "002074000", "000530180",
           "056800000", "230000009", "600025000", "008769230"],   # Printable Puzzles, Medium 1
}
LABELS = ["TL", "TR", "BR", "BL"]
VIEW_MAX = 900     # genel görünümün en uzun kenarı (piksel)
ZOOM_HALF = 60     # büyütme penceresi: tıklanan noktanın etrafında ±60 orijinal piksel
ZOOM = 6           # büyütme katsayısı
VALID_CONDITIONS = {"C1", "C2", "C3", "C4", "C5"}


def pick_corners(img):
    """Kullanıcıya 4 köşeyi tıklatır, orijinal görüntü koordinatlarında döndürür."""
    h, w = img.shape[:2]
    s = VIEW_MAX / max(h, w)
    view = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    corners, clicks = [], []

    def on_view(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Görünümdeki piksel merkezini orijinal piksel koordinatına çevir (OpenCV kuralı: piksel merkezi = tam sayı)
            clicks.append(((x + 0.5) / s - 0.5, (y + 0.5) / s - 0.5))

    cv2.namedWindow("genel")
    cv2.setMouseCallback("genel", on_view)
    while len(corners) < 4:
        shown = view.copy()
        for i, (cx, cy) in enumerate(corners):
            p = (int((cx + 0.5) * s), int((cy + 0.5) * s))
            cv2.circle(shown, p, 5, (0, 0, 255), -1)
            cv2.putText(shown, LABELS[i], (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(shown, f"Tikla: {LABELS[len(corners)]}  (r: sifirla, ESC: cik)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("genel", shown)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            sys.exit("Çıkıldı, kaydedilmedi.")
        if key == ord("r"):
            corners.clear()
        if clicks:
            corners.append(refine(img, *clicks.pop()))
    return corners


def refine(img, x, y):
    """Tıklanan noktanın çevresini büyütüp hassas köşeyi seçtirir."""
    h, w = img.shape[:2]
    x0 = int(max(0, min(w - 2 * ZOOM_HALF, round(x) - ZOOM_HALF)))
    y0 = int(max(0, min(h - 2 * ZOOM_HALF, round(y) - ZOOM_HALF)))
    crop = img[y0:y0 + 2 * ZOOM_HALF, x0:x0 + 2 * ZOOM_HALF]
    big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    result = []

    def on_zoom(event, zx, zy, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            result.append((x0 + (zx + 0.5) / ZOOM - 0.5, y0 + (zy + 0.5) / ZOOM - 0.5))

    cv2.namedWindow("buyutme")
    cv2.setMouseCallback("buyutme", on_zoom)
    while not result:
        cv2.imshow("buyutme", big)
        if (cv2.waitKey(30) & 0xFF) == 27:   # ESC: ilk tıklamayı olduğu gibi kabul et
            result.append((x, y))
    cv2.destroyWindow("buyutme")
    return result[0]


def confirm(img, corners):
    """Seçilen dörtgeni gösterir; 'y' onay, başka tuş red."""
    h, w = img.shape[:2]
    s = VIEW_MAX / max(h, w)
    shown = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    pts = [(int((cx + 0.5) * s), int((cy + 0.5) * s)) for cx, cy in corners]
    for i in range(4):
        cv2.line(shown, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
        cv2.putText(shown, LABELS[i], pts[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(shown, "y: kaydet   baska tus: tekrar", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("genel", shown)
    return (cv2.waitKey(0) & 0xFF) == ord("y")


def ask_puzzle():
    while True:
        key = input("Bulmaca (P1 / P2 / P3): ").strip().upper()
        if key in PUZZLES:
            return [[int(c) for c in row] for row in PUZZLES[key]]
        print("Geçersiz, tekrar dene.")


def ask_conditions():
    while True:
        conds = input("Koşullar (ör. C2 C5): ").upper().split()
        if conds and set(conds) <= VALID_CONDITIONS:
            return sorted(set(conds))
        print("Geçersiz, sadece C1-C5 kullan.")


def main():
    if len(sys.argv) != 2:
        sys.exit("Kullanım: python annotate.py <fotoğraf yolu>")
    img_path = sys.argv[1]
    json_path = os.path.splitext(img_path)[0] + ".json"
    if os.path.exists(json_path) and input(f"{json_path} zaten var, üzerine yazılsın mı? (e/h): ").lower() != "e":
        sys.exit("İptal edildi.")
    img = cv2.imread(img_path)   # EXIF yönünü uygular; programın göreceği görüntüyle aynı
    if img is None:
        sys.exit(f"Fotoğraf okunamadı: {img_path}")

    while True:
        corners = pick_corners(img)
        if confirm(img, corners):
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    data = {
        "image": os.path.basename(img_path),
        "conditions": ask_conditions(),
        "corners": [[round(x, 2), round(y, 2)] for x, y in corners],
        "grid": ask_puzzle(),
    }
    with open(json_path, "w") as f:
        f.write("{\n")
        f.write(f'  "image": {json.dumps(data["image"])},\n')
        f.write(f'  "conditions": {json.dumps(data["conditions"])},\n')
        f.write(f'  "corners": {json.dumps(data["corners"])},\n')
        f.write('  "grid": [\n')
        f.write(",\n".join("    " + json.dumps(r) for r in data["grid"]))
        f.write("\n  ]\n}\n")
    print(f"Kaydedildi: {json_path}")


if __name__ == "__main__":
    main()
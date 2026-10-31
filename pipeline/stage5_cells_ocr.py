"""Stage 5: Hücrelere bölme, boş/dolu kararı ve rakam tanıma.

Girdi: Stage 3'ün H4 rektifikasyonu (SIZE x SIZE, hücre başına CELL piksel).
Adımlar:
    1. 81 hücreye eşit bölme; ızgara çizgileri OCR'a girmesin diye her hücrenin kenarından pay kesilir.
    2. Boş/dolu kararı, OCR'dan ÖNCE ve açık bir ölçütle: hücredeki en büyük mürekkep lekesinin alanı.
    3. Dolu hücreler OCR için hazırlanır: ikileştirme, kenar kalıntılarını atma, rakamı ortalama,
       beyaz pay ekleme, büyütme.
    4. Tesseract önce tek satır/ham modda (--psm 13) sadece 1-9 rakamlarına izin vererek okur;
       boş dönerse tek karakter modunda (--psm 10) yeniden dener.
"""
import cv2
import numpy as np
import pytesseract

from pipeline.stage3_rectify import CELL

MARGIN_FRAC = 0.12      # hücre kenarından kesilen pay (her yandan) = hücrenin %12'si
MIN_INK_FRAC = 0.03     # en büyük leke kırpılmış hücre alanının en az %3'ü olmalı → dolu
MIN_HEIGHT_FRAC = 0.30  # ... ve yüksekliği kırpılmış hücrenin en az %30'u olmalı (nokta/gürültü değil)
OCR_HEIGHT = 32         # hazırlanmış rakamın yüksekliği (piksel); 32 ve 48 denendi, 32 daha iyi
OCR_PAD = 16            # rakamın etrafına eklenen beyaz pay (piksel)
# Tesseract ayarları (deneme setinde 565 dolu hücre üzerinde karşılaştırıldı):
#   --psm 10 + 1-9 beyaz listesi : %88.7  (LSTM bu modda beyaz listeyle sık sık boş dönüyor)
#   --psm 8  + beyaz liste       : %95.3
#   --psm 13 + beyaz liste       : %96.1  ← ana ayar
#   --psm 10, beyaz liste yok    : %95.0
#   13 (beyaz liste) boş dönerse 10 (beyaz listesiz) dene : %96.8  ← kullanılan
TESS_PRIMARY = "--psm 13 --oem 1 -c tessedit_char_whitelist=123456789"
TESS_FALLBACK = "--psm 10 --oem 1"


def cell_crops(rect_gray):
    """81 hücreyi (9x9 liste) kenar payı kesilmiş olarak döndürür."""
    m = int(round(MARGIN_FRAC * CELL))
    cells = []
    for r in range(9):
        row = []
        for c in range(9):
            y0, x0 = r * CELL, c * CELL
            row.append(rect_gray[y0 + m:y0 + CELL - m, x0 + m:x0 + CELL - m])
        cells.append(row)
    return cells


def binarize(cell):
    """Mürekkep = 255, kâğıt = 0. Otsu hücre içinde ayrı ayrı uygulanır (yerel ışığa uyum)."""
    _, b = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Boş bir hücrede Otsu kâğıt dokusunu "mürekkep" sanabilir; çok açık bir hücreyi boş kabul et
    if cell.max() - cell.min() < 40:
        b[:] = 0
    return b


def largest_blob(binary):
    """Kenara değmeyen en büyük bağlı bileşeni döndürür (maske, alan, kutu) ya da None.

    Kenara değen lekeler büyük ihtimalle ızgara çizgisi kalıntısıdır ve atılır.
    """
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches and (bw > 0.8 * w or bh > 0.8 * h):   # uzun ince kenar kalıntısı
            continue
        if best is None or area > stats[best][4]:
            best = i
    if best is None:
        return None
    x, y, bw, bh, area = stats[best]
    return (labels == best).astype(np.uint8) * 255, int(area), (x, y, bw, bh)


def is_filled(cell):
    """Boş/dolu kararı: en büyük mürekkep lekesi yeterince büyük ve uzun mu?"""
    blob = largest_blob(binarize(cell))
    if blob is None:
        return False
    _, area, (_, _, _, bh) = blob
    h, w = cell.shape
    return area >= MIN_INK_FRAC * h * w and bh >= MIN_HEIGHT_FRAC * h


def prepare_cell(cell):
    """OCR için hücre hazırlığı: ikileştir, sadece rakam lekesini tut, ortala, pay ekle, büyüt.

    Tesseract koyu yazı / açık zemin bekler, bu yüzden sonuç siyah rakam beyaz zemindir.
    """
    blob = largest_blob(binarize(cell))
    if blob is None:
        return None
    mask, _, (x, y, bw, bh) = blob
    digit = mask[y:y + bh, x:x + bw]
    scale = OCR_HEIGHT / bh
    digit = cv2.resize(digit, (max(1, round(bw * scale)), OCR_HEIGHT), interpolation=cv2.INTER_CUBIC)
    digit = cv2.copyMakeBorder(digit, OCR_PAD, OCR_PAD, OCR_PAD, OCR_PAD, cv2.BORDER_CONSTANT, value=0)
    return 255 - digit


def prepare_raw(cell):
    """Karşılaştırma için 'hazırlıksız' hücre: sadece büyütme, başka işlem yok."""
    return cv2.resize(cell, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def first_digit(text):
    """Metindeki ilk 1-9 rakamını döndürür, yoksa 0."""
    for ch in text:
        if ch in "123456789":
            return int(ch)
    return 0


def ocr_digit(img):
    """Tek bir hücre görüntüsünü okur. Önce beyaz listeli ana ayar, boş dönerse yedek ayar."""
    d = first_digit(pytesseract.image_to_string(img, config=TESS_PRIMARY))
    if d == 0:
        d = first_digit(pytesseract.image_to_string(img, config=TESS_FALLBACK))
    return d


def recognize_grid(rect_bgr, prepare=True):
    """H4 görüntüsünden 9x9 grid okur.

    prepare=False: hazırlıksız hücreler (rapor için 'önce' durumu).
    Dönüş: (grid, filled) — grid 9x9 int liste, filled 9x9 bool liste.
    """
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY) if rect_bgr.ndim == 3 else rect_bgr
    cells = cell_crops(gray)
    grid = [[0] * 9 for _ in range(9)]
    filled = [[False] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            if not is_filled(cells[r][c]):
                continue
            filled[r][c] = True
            img = prepare_cell(cells[r][c]) if prepare else prepare_raw(cells[r][c])
            if img is not None:
                grid[r][c] = ocr_digit(img)
    return grid, filled
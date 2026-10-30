"""Stage 1: Ön işleme.

Adımlar:
    1. Griye çevir.
    2. Büyük fotoğrafı sabit bir çalışma boyutuna küçült (hız ve ölçek bağımsızlığı için).
    3. Gürültüyü azalt (Gauss bulanıklığı).
    4. Düzensiz aydınlatmayı dengele: görüntüyü bulanık "arka plan" tahminine böl.

Bütün çekirdek boyutları görüntü boyutunun bir oranı olarak verilir, böylece aynı kod
1000 px ve 4000 px fotoğrafta aynı şekilde çalışır.
"""
import cv2
import numpy as np

WORK_SIZE = 1000          # küçültülmüş görüntünün en uzun kenarı (piksel)
BLUR_FRAC = 0.002         # gürültü azaltma çekirdeği = genişliğin %0.2'si (1000 px'te ~3 px)
BACKGROUND_FRAC = 0.05    # arka plan tahmini çekirdeği = genişliğin %5'i (1000 px'te ~51 px)


def odd(n):
    """Çekirdek boyutları tek sayı olmalı (merkez pikseli olsun diye)."""
    n = max(3, int(round(n)))
    return n if n % 2 == 1 else n + 1


def preprocess(img):
    """Renkli fotoğrafı Harris için hazırlar.

    Dönüş:
        norm  : aydınlatması dengelenmiş gri görüntü (uint8, küçültülmüş)
        scale : küçültme oranı. Küçük görüntüdeki (x, y) -> orijinalde ((x+0.5)/scale - 0.5, ...)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Küçült (sadece büyütme yapma)
    h, w = gray.shape
    scale = min(1.0, WORK_SIZE / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    width = gray.shape[1]

    # 3. Gürültü azaltma
    k = odd(BLUR_FRAC * width)
    smooth = cv2.GaussianBlur(gray, (k, k), 0)

    # 4. Aydınlatma dengeleme: kâğıdın yerel parlaklığını tahmin et ve ona böl.
    #    Median bulanıklığı ince çizgileri ve rakamları siler, geriye yavaş değişen ışık kalır.
    #    Bölünce gölgeli ve aydınlık bölgeler aynı beyaz seviyesine gelir; mürekkep koyu kalır.
    kb = odd(BACKGROUND_FRAC * width)
    background = cv2.medianBlur(smooth, kb)
    norm = cv2.divide(smooth, background, scale=255)

    return norm, scale


def to_original(points, scale):
    """Küçük görüntüdeki noktaları orijinal fotoğraf koordinatlarına çevirir."""
    pts = np.asarray(points, dtype=np.float64)
    return (pts + 0.5) / scale - 0.5
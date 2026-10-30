"""Stage 3: Afin ve projektif modellerle rektifikasyon.

Bulunan 4 köşeyi 450x450 piksellik bir kareye (hücre başına 50 px) eşler ve üç model üretir:
    A3: afin, 3 nokta   (cv2.getAffineTransform)  -> cv2.warpAffine
    A4: afin, 4 nokta   (cv2.estimateAffine2D, en küçük kareler) -> cv2.warpAffine
    H4: homografi       (cv2.getPerspectiveTransform) -> cv2.warpPerspective

Köşe sırası her yerde: TL, TR, BR, BL (indeks 0, 1, 2, 3).
"""
import cv2
import numpy as np

CELL = 50                 # hücre boyutu (piksel). Ör. 64 yapılırsa hedef 576x576 olur
SIZE = 9 * CELL           # hedef karenin kenarı: 450

# A3'te kullanılan üç köşenin indeksleri. Varsayılan TL, TR, BL.
# Ör. TL, TR, BR kullanmak için (0, 1, 2) yap; o zaman kullanılmayan köşe BL olur.
A3_CORNERS = (0, 1, 3)

# Hedef karenin köşeleri. OpenCV'de piksel merkezi tam sayıdır, piksel kenarı ise -0.5'tedir.
# Köşeleri kenara (-0.5 ve SIZE-0.5) eşlersek her hücre tam olarak CELL x CELL piksel olur.
TARGET = np.float32([
    [-0.5, -0.5],                 # TL
    [SIZE - 0.5, -0.5],           # TR
    [SIZE - 0.5, SIZE - 0.5],     # BR
    [-0.5, SIZE - 0.5],           # BL
])

WHITE = (255, 255, 255)           # görüntü dışına düşen bölgeler beyaz dolsun (OCR için daha iyi)


def apply_affine(M, pts):
    """2x3 afin matrisi noktalara uygular: [x', y'] = M @ [x, y, 1]."""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    return np.hstack([pts, ones]) @ M.T


def estimate_a3(src):
    """3 nokta çiftinden afin dönüşüm ve kullanılmayan 4. köşenin hatası."""
    idx = list(A3_CORNERS)
    A3 = cv2.getAffineTransform(src[idx], TARGET[idx])
    unused = [i for i in range(4) if i not in idx][0]
    landed = apply_affine(A3, src[[unused]])[0]           # 4. köşe hedefte nereye düştü
    err = float(np.linalg.norm(landed - TARGET[unused]))  # ideal yerine uzaklığı (piksel)
    return A3, landed, err


def estimate_a4(src):
    """4 nokta çiftinden en küçük kareler afin dönüşümü ve RMS artık hatası.

    estimateAffine2D varsayılan olarak RANSAC kullanır ve bazı noktaları "aykırı" sayıp atabilir.
    Eşiği çok büyük verince dört nokta da içeride kalır; ardından gelen Levenberg-Marquardt
    iyileştirmesi kare hataların toplamını en aza indirir, yani sonuç en küçük kareler çözümüdür.
    """
    A4, _ = cv2.estimateAffine2D(src, TARGET, method=cv2.RANSAC,
                                 ransacReprojThreshold=1e6, refineIters=50)
    residuals = apply_affine(A4, src) - TARGET            # her köşenin artık vektörü
    rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    return A4, rms


def estimate_h4(src):
    """4 nokta çiftinden homografi ve köşelerin hedefe ne kadar tam oturduğu."""
    H4 = cv2.getPerspectiveTransform(src, TARGET)
    mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H4).reshape(-1, 2)
    max_err = float(np.max(np.linalg.norm(mapped - TARGET, axis=1)))   # ~0 olmalı
    return H4, max_err


def rectify(img, corners):
    """Üç modeli hesaplar ve üç düzeltilmiş görüntüyü üretir.

    corners: orijinal görüntüde [[x, y], ...] TL, TR, BR, BL sırasıyla.
    Dönen sözlükteki matrisler ve hatalar result.json'a yazılacak.
    """
    src = np.float32(corners)
    A3, a3_landed, a3_err = estimate_a3(src)
    A4, a4_rms = estimate_a4(src)
    H4, h4_err = estimate_h4(src)

    size = (SIZE, SIZE)
    flags = cv2.INTER_LINEAR
    return {
        "A3": A3, "A4": A4, "H4": H4,
        "a3_corner_err_px": a3_err,
        "a3_landed": a3_landed,
        "a4_rms_px": a4_rms,
        "h4_max_err_px": h4_err,
        "rect_A3": cv2.warpAffine(img, A3, size, flags=flags, borderValue=WHITE),
        "rect_A4": cv2.warpAffine(img, A4, size, flags=flags, borderValue=WHITE),
        "rect_H4": cv2.warpPerspective(img, H4, size, flags=flags, borderValue=WHITE),
    }
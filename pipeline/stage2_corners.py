"""Stage 2: Harris kose tespiti."""

import cv2
import numpy as np

# Harris parametreleri
HARRIS_BLOCK = 5
HARRIS_KSIZE = 3
HARRIS_K = 0.04
THR = 0.01
NMS_FRAC = 0.01


def harris_response(gray):
    """Harris yanit haritasini hesaplar."""

    gray = np.float32(gray)

    R = cv2.cornerHarris(
        gray, HARRIS_BLOCK, HARRIS_KSIZE, HARRIS_K
    )

    return R


def find_candidates(R):
    """Harris haritasindan aday koseleri bulur."""

    width = R.shape[1]

    # Komsuluk boyutu
    n = max(3, round(width * NMS_FRAC))

    if n % 2 == 0:
        n += 1

    # Yerel maksimumlari bul
    kernel = np.ones((n, n), np.uint8)
    max_R = cv2.dilate(R, kernel)

    # Guclu koseleri sec
    corners = (R == max_R) & (R > THR * R.max())

    # Kose koordinatlarini al
    points = np.argwhere(corners)

    candidates = []

    for y, x in points:
        candidates.append((x, y, R[y, x]))

    return candidates


def find_puzzle_region(norm):
    """Goruntudeki en buyuk dortgeni bulur."""

    # Goruntuyu gri yap
    if len(norm.shape) == 3:
        gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    else:
        gray = norm.copy()

    gray = gray.astype(np.uint8)

    # Siyah beyaz goruntu
    _, binary = cv2.threshold(
        gray, 200, 255,
        cv2.THRESH_BINARY_INV 
    )

        # Hata ayiklama: siyah beyaz goruntuyu kaydet (gecici)
    cv2.imwrite("results/debug_binary.png", binary)

    # Konturlari bul
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = gray.shape
    min_area = height * width * 0.02

    biggest = None
    biggest_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # Konturu dortgene yaklastir
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)

        approx = cv2.approxPolyDP(
            hull, 0.02 * perimeter, True
        )

        # Dort koseli ve disbukey olmali
        if len(approx) == 4 and cv2.isContourConvex(approx):

            if area > biggest_area:
                biggest_area = area
                biggest = approx

    if biggest is None:
        return None

    return biggest.reshape(4, 2)
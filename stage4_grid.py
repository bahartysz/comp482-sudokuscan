
"""Stage 4: Izgara cizgilerinin sapmasini hesaplama."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

import json
from stage4_compare import calculate_tilt


def calculate_grid_error(profile):
    """Izgara cizgilerinin ideal konumdan sapmasini bulur."""

    errors = []
    positions = []

    # Ideal cizgi konumlari: 0, 50, 100, ..., 450
    for ideal in range(50, 401, 50):

        # Ideal konumun 15 piksel cevresine bak
        start = max(0, ideal - 15)
        end = min(len(profile), ideal + 16)

        # En yuksek degerin oldugu konumu bul
        part = profile[start:end]
        real = start + np.argmax(part)

        # Gercek ve ideal konum arasindaki fark
        error = abs(real - ideal)

        errors.append(error)
        positions.append(real)

    # Ortalama sapma
    mean_error = np.mean(errors)

    return mean_error, positions



if __name__ == "__main__":

    # Tum fotograflarin sonuc dosyalarini bul
    files = glob.glob("results/personal/*_result.json")

    for yol in sorted(files):

        # JSON dosyasini oku
        with open(yol, "r") as f:
            result = json.load(f)

        if result["verdict"] == "not_found":
            continue

        # Egiklik oranini hesapla
        tilt = calculate_tilt(result["corners"])

        # Fotograf adini al
        name = os.path.basename(yol).replace("_result.json", "")

        # A3, A4 ve H4 sonuclari
        errors = {}

        for model in ["A3", "A4", "H4"]:

            image_path = yol.replace(
                "_result.json", f"_rect_{model}.png"
            )

            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if gray is None:
                errors[model] = None
                continue

            # Koyu cizgileri belirginlestir
            inverse = 255 - gray

            # Dikey ve yatay profiller
            vertical = np.sum(inverse, axis=0)
            horizontal = np.sum(inverse, axis=1)

            # Sapmalari hesapla
            v_error, _ = calculate_grid_error(vertical)
            h_error, _ = calculate_grid_error(horizontal)

            errors[model] = (v_error + h_error) / 2

        # Sonuclari yazdir
        print(
            name,
            "Egiklik:", round(tilt, 3),
            "A3:", round(errors["A3"], 2) if errors["A3"] is not None else "yok",
            "A4:", round(errors["A4"], 2) if errors["A4"] is not None else "yok",
            "H4:", round(errors["H4"], 2) if errors["H4"] is not None else "yok"
        )

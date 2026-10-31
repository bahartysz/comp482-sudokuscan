
"""Stage 4: Izgara cizgilerinin sapmasini hesaplama."""

import cv2
import numpy as np
import glob
import json
import os

from stage4_compare import calculate_tilt


def calculate_grid_error(profile):
    """Izgara cizgilerinin ideal konumdan sapmasini bulur."""

    errors = []
    positions = []

    # Ideal ic cizgi konumlari
    for ideal in range(50, 401, 50):

        # 15 piksel cevresinde ara
        start = max(0, ideal - 15)
        end = min(len(profile), ideal + 16)

        # En yuksek degeri bul
        part = profile[start:end]
        real = start + np.argmax(part)

        # Sapmayi hesapla
        error = abs(real - ideal)

        errors.append(error)
        positions.append(real)

    mean_error = np.mean(errors)

    # Ortalama, konumlar ve en buyuk sapma
    return mean_error, positions, max(errors)


if __name__ == "__main__":

    # Tum fotograflarin sonuc dosyalari
    files = glob.glob("results/personal/*_result.json")

    for yol in sorted(files):

        with open(yol, "r") as f:
            result = json.load(f)

        if result["verdict"] == "not_found":
            continue

        # Egiklik hesapla
        tilt = calculate_tilt(result["corners"])

        # Fotograf adi
        name = os.path.basename(yol).replace("_result.json", "")

        errors = {}
        max_errors = {}

        # Uc donusumu de kontrol et
        for model in ["A3", "A4", "H4"]:

            image_path = yol.replace(
                "_result.json", f"_rect_{model}.png"
            )

            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if gray is None:
                errors[model] = None
                max_errors[model] = None
                continue

            # Koyu cizgileri belirginlestir
            inverse = 255 - gray

            # Dikey ve yatay profiller
            vertical = np.sum(inverse, axis=0)
            horizontal = np.sum(inverse, axis=1)

            # Ortalama ve maksimum sapmalar
            v_error, _, v_max = calculate_grid_error(vertical)
            h_error, _, h_max = calculate_grid_error(horizontal)

            errors[model] = (v_error + h_error) / 2
            max_errors[model] = max(v_max, h_max)

        # Sonuclari yazdir
        print(name, "Egiklik:", round(tilt, 3), end="  ")

        for model in ["A3", "A4", "H4"]:

            if errors[model] is None:
                print(model + ": yok", end="  ")
            else:
                print(
                    model + ":",
                    round(errors[model], 2),
                    "Max:", max_errors[model],
                    end="  "
                )

        print()

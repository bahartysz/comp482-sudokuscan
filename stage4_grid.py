
#Stage 4: Izgara cizgilerinin sapmasini hesaplama.

import cv2
import numpy as np
import glob
import json
import os
import matplotlib.pyplot as plt

from stage4_compare import calculate_tilt


def calculate_grid_error(profile):
    """Izgara cizgilerinin ideal konumdan sapmasini bulur."""

    errors = []
    positions = []

    # Ideal ic cizgi konumlari
    for ideal in range(50, 401, 50):

        # Ideal cizginin 24 piksel cevresinde ara
        start = max(0, ideal - 24)
        end = min(len(profile), ideal + 25)

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

    # Grafik icin listeler
    tilts = []
    a3_values = []
    a4_values = []
    h4_values = []

    for yol in sorted(files):

        # JSON dosyasini oku
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

        # A3, A4 ve H4 goruntulerini incele
        for model in ["A3", "A4", "H4"]:

            image_path = yol.replace(
                "_result.json", f"_rect_{model}.png"
            )

            # Goruntuyu gri oku
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if gray is None:
                errors[model] = None
                max_errors[model] = None
                continue

            # Koyu cizgileri belirginlestir
            inverse = 255 - gray

            # Dikey ve yatay profilleri hesapla
            vertical = np.sum(inverse, axis=0)
            horizontal = np.sum(inverse, axis=1)

            # Ortalama ve maksimum sapmalari bul
            v_error, _, v_max = calculate_grid_error(vertical)
            h_error, _, h_max = calculate_grid_error(horizontal)

            errors[model] = (v_error + h_error) / 2
            max_errors[model] = max(v_max, h_max)

        # Terminale sonuclari yazdir
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

        # Grafik icin degerleri listelere ekle
        if all(errors[m] is not None for m in ["A3", "A4", "H4"]):

            tilts.append(tilt)
            a3_values.append(errors["A3"])
            a4_values.append(errors["A4"])
            h4_values.append(errors["H4"])

    # Grafik ciz
    if len(tilts) > 0:

        plt.figure(figsize=(9, 5))

        plt.scatter(tilts, a3_values, label="A3")
        plt.scatter(tilts, a4_values, label="A4")
        plt.scatter(tilts, h4_values, label="H4")

        plt.xlabel("Eğiklik oranı")
        plt.ylabel("Ortalama ızgara sapması (piksel)")
        plt.title("Eğiklik ve Izgara Çizgisi Sapması (8 İç Çizgi)")

        plt.legend()
        plt.grid(True)

        # Grafigi kaydet
        os.makedirs("rapor_gorselleri", exist_ok=True)

        plt.savefig(
            "rapor_gorselleri/stage4_izgara_sapmasi.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print("Izgara sapmasi grafigi kaydedildi.")

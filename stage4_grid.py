
"""Stage 4: Izgara cizgilerinin sapmasini hesaplama."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def calculate_grid_error(profile):
    """Izgara cizgilerinin ideal konumdan sapmasini bulur."""

    errors = []
    positions = []

    # Ideal cizgi konumlari: 0, 50, 100, ..., 450
    for ideal in range(0, 451, 50):

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

    # Ilk fotografimizin A3 goruntusunu bul
    files = glob.glob("results/personal/*_01_rect_A3.png")

    if len(files) == 0:
        print("Goruntu bulunamadi.")

    else:
        yol = files[0]

        # Goruntuyu gri olarak oku
        gray = cv2.imread(yol, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            print("Goruntu okunamadi.")

        else:
            # Siyah cizgileri buyuk deger yap
            inverse = 255 - gray

            # Dikey ve yatay profiller
            vertical = np.sum(inverse, axis=0)
            horizontal = np.sum(inverse, axis=1)

            # Sapmalari hesapla
            v_error, v_positions = calculate_grid_error(vertical)
            h_error, h_positions = calculate_grid_error(horizontal)

            total_error = (v_error + h_error) / 2

            print("Dikey cizgiler:", v_positions)
            print("Yatay cizgiler:", h_positions)

            print("Dikey sapma:", round(v_error, 2))
            print("Yatay sapma:", round(h_error, 2))
            print("Ortalama sapma:", round(total_error, 2))

            # Grafik icin klasor
            os.makedirs("rapor_gorselleri", exist_ok=True)

            # Dikey cizgi profili
            plt.figure(figsize=(9, 4))
            plt.plot(vertical)

            plt.xlabel("Sutun konumu (piksel)")
            plt.ylabel("Koyu piksel toplami")
            plt.title("A3 Dikey Izgara Profili - 01")
            plt.grid(True)

            plt.savefig(
                "rapor_gorselleri/stage4_01_dikey_profil.png",
                dpi=200,
                bbox_inches="tight"
            )

            plt.close()

            # Yatay cizgi profili
            plt.figure(figsize=(9, 4))
            plt.plot(horizontal)

            plt.xlabel("Satir konumu (piksel)")
            plt.ylabel("Koyu piksel toplami")
            plt.title("A3 Yatay Izgara Profili - 01")
            plt.grid(True)

            plt.savefig(
                "rapor_gorselleri/stage4_01_yatay_profil.png",
                dpi=200,
                bbox_inches="tight"
            )

            plt.close()

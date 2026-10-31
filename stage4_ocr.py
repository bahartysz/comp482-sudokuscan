
"""Stage 4: A3, A4 ve H4 OCR dogruluklarini karsilastirma."""

import cv2
import glob
import json
import os
import matplotlib.pyplot as plt

from stage4_compare import calculate_tilt
from pipeline.stage5_cells_ocr import recognize_grid


def calculate_accuracy(grid, correct_grid):
    """81 hucreden kacinin dogru okundugunu hesaplar."""

    correct = 0

    for row in range(9):
        for col in range(9):
            if grid[row][col] == correct_grid[row][col]:
                correct += 1

    accuracy = correct / 81 * 100

    return accuracy


if __name__ == "__main__":

    # Tum fotograflarin sonuc dosyalarini bul
    files = glob.glob("results/personal/*_result.json")

    # Grafik icin listeler
    tilts = []
    a3_values = []
    a4_values = []
    h4_values = []

    for yol in sorted(files):

        with open(yol, "r") as f:
            result = json.load(f)

        if result["verdict"] == "not_found":
            continue

        name = os.path.basename(yol).replace("_result.json", "")

        # Egiklik oranini hesapla
        tilt = calculate_tilt(result["corners"])

        # Dogru Sudoku etiketini oku
        label_path = "data/personal/" + name + ".json"

        if not os.path.exists(label_path):
            print(name, "Etiket dosyasi bulunamadi.")
            continue

        with open(label_path, "r") as f:
            label = json.load(f)

        correct_grid = label["grid"]

        accuracies = {}

        # Uc donusumun OCR sonuclarini hesapla
        for model in ["A3", "A4", "H4"]:

            image_path = yol.replace(
                "_result.json", f"_rect_{model}.png"
            )

            # Goruntuyu renkli oku
            image = cv2.imread(image_path)

            if image is None:
                accuracies[model] = None
                continue

            # OCR ile rakamlari oku
            grid, filled = recognize_grid(image)

            # Dogru hucre oranini hesapla
            accuracy = calculate_accuracy(grid, correct_grid)

            accuracies[model] = accuracy

        # Sonuclari yazdir
        print(name, "Egiklik:", round(tilt, 3), end="  ")

        for model in ["A3", "A4", "H4"]:
            if accuracies[model] is None:
                print(model + ": yok", end="  ")
            else:
                print(
                    model + ":",
                    round(accuracies[model], 2),
                    "%",
                    end="  "
                )

        print()

        # Grafik icin degerleri ekle
        if all(accuracies[m] is not None for m in ["A3", "A4", "H4"]):

            tilts.append(tilt)
            a3_values.append(accuracies["A3"])
            a4_values.append(accuracies["A4"])
            h4_values.append(accuracies["H4"])

    # OCR dogruluk grafigi
    if len(tilts) > 0:

        plt.figure(figsize=(9, 5))

        plt.scatter(tilts, a3_values, label="A3")
        plt.scatter(tilts, a4_values, label="A4")
        plt.scatter(tilts, h4_values, label="H4")

        # %95 dogruluk esigi
        plt.axhline(
            95,
            linestyle="--",
            label="%95 dogruluk esigi"
        )

        plt.xlabel("Eğiklik oranı")
        plt.ylabel("Hücre doğruluğu (%)")
        plt.title("Eğiklik ve OCR Doğruluğu")

        plt.legend()
        plt.grid(True)

        # Grafigi kaydet
        os.makedirs("rapor_gorselleri", exist_ok=True)

        plt.savefig(
            "rapor_gorselleri/stage4_ocr_dogrulugu.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print("OCR dogruluk grafigi kaydedildi.")

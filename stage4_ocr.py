
"""Stage 4: A3, A4 ve H4 OCR dogruluklarini karsilastirma."""

import cv2
import glob
import json
import os

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

    for yol in sorted(files):

        # Sonuc JSON dosyasini oku
        with open(yol, "r") as f:
            result = json.load(f)

        if result["verdict"] == "not_found":
            continue

        # Fotograf adi
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

        # A3, A4 ve H4 dogruluklari
        accuracies = {}

        for model in ["A3", "A4", "H4"]:

            image_path = yol.replace(
                "_result.json", f"_rect_{model}.png"
            )

            # Goruntuyu renkli oku
            image = cv2.imread(image_path)

            if image is None:
                accuracies[model] = None
                continue

            # OCR ile Sudoku'yu oku
            grid, filled = recognize_grid(image)

            # Dogru cevapla karsilastir
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

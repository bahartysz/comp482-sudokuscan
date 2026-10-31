
"""Stage 4: Egiklik ve kose hatalarini karsilastirma."""

import numpy as np
import glob
import json
import os


def calculate_tilt(corners):
    """Ust ve alt kenar uzunluklarindan egikligi hesaplar."""

    # Koseler TL, TR, BR, BL sirasinda
    TL, TR, BR, BL = corners

    # Ust kenar uzunlugu
    top = np.linalg.norm(
        np.array(TR) - np.array(TL)
    )

    # Alt kenar uzunlugu
    bottom = np.linalg.norm(
        np.array(BR) - np.array(BL)
    )

    # Kenar uzunluklari orani
    tilt = max(top, bottom) / min(top, bottom)

    return tilt


if __name__ == "__main__":

    # Sonuc dosyalarini bul
    files = glob.glob("results/personal/*_result.json")

    # Her fotograf icin degerleri hesapla
    for yol in sorted(files):

        # JSON dosyasini oku
        with open(yol, "r") as f:
            result = json.load(f)

        # Kose bulunamayan fotograflari atla
        if result["verdict"] == "not_found":
            continue

        # Dort koseyi al
        corners = result["corners"]

        # Egiklik oranini hesapla
        tilt = calculate_tilt(corners)

        # A3 ve A4 hatalarini al
        a3_error = result["a3_corner_err_px"]
        a4_error = result["a4_rms_px"]

        # Dosya adini al
        name = os.path.basename(yol)

        # Sonuclari yazdir
        print(
            name,
            "Egiklik:", round(tilt, 3),
            "A3 hata:", round(a3_error, 2),
            "A4 RMS:", round(a4_error, 2)
        )

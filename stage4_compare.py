import numpy as np

def calculate_tilt(corners):
    """Ust ve alt kenar uzunluklarindan egikligi hesaplar."""

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

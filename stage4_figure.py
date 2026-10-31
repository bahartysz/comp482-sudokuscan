"""Stage 4: _09 fotografi icin A3, A4 ve H4 goruntulerini yan yana goster."""

import cv2
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Dosya yolları
    path_a3 = "results/personal/s332501004_09_rect_A3.png"
    path_a4 = "results/personal/s332501004_09_rect_A4.png"
    path_h4 = "results/personal/s332501004_09_rect_H4.png"

    # Görüntüleri oku
    img_a3 = cv2.imread(path_a3)
    img_a4 = cv2.imread(path_a4)
    img_h4 = cv2.imread(path_h4)

    # Kontrol
    if img_a3 is None or img_a4 is None or img_h4 is None:
        print("Goruntulerden en az biri okunamadi.")
    else:
        # BGR -> RGB cevir
        img_a3 = cv2.cvtColor(img_a3, cv2.COLOR_BGR2RGB)
        img_a4 = cv2.cvtColor(img_a4, cv2.COLOR_BGR2RGB)
        img_h4 = cv2.cvtColor(img_h4, cv2.COLOR_BGR2RGB)

        # 3'lü şekil
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        images = [img_a3, img_a4, img_h4]
        titles = ["A3 (OCR %58.0)", "A4 (OCR %66.7)", "H4 (OCR %92.6)"]

        for i in range(3):
            axes[i].imshow(images[i])
            axes[i].set_title(titles[i])
            axes[i].axis("off")

            # Ideal izgara cizgileri
            for pos in range(0, 451, 50):
                axes[i].axhline(pos, color="red", linewidth=0.5)
                axes[i].axvline(pos, color="red", linewidth=0.5)

            # Goruntunun disina tasmayi engelle
            axes[i].set_xlim(-0.5, 449.5)
            axes[i].set_ylim(449.5, -0.5)

        plt.tight_layout()

        # Klasor yoksa olustur
        os.makedirs("rapor_gorselleri", exist_ok=True)

        # Kaydet
        plt.savefig(
            "rapor_gorselleri/stage4_09_yan_yana.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print("Yan yana sekil kaydedildi.")
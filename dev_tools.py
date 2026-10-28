import json
import sys
import cv2

# 1. Fotoğraf adını komut satırından almak için
name = sys.argv[1]

# 2. Dosya yollarını oluşturma
img_path = f"data/dev/images/{name}.jpg"
json_path = f"data/dev/annotations/{name}.json"

# 3. JSON dosyasını açma ve okuma
with open(json_path) as f:
    data = json.load(f)
corners = data["corners"]

# 4. Fotoğrafı okuma
img = cv2.imread(img_path)
if img is None:
    sys.exit(f"Fotoğraf okunamadı: {img_path}")

# 5. Daire yarıçapını görüntü genişliğine göre ayarlama
width = img.shape[1]
radius = max(3, int(width * 0.01))
thickness = max(1, int(width * 0.002))
font_scale = width / 1000

# 6. Her köşeye kırmızı dolu daire ve etiket çizme
labels = ["TL", "TR", "BR", "BL"]
for label, (x, y) in zip(labels, corners):
    point = (int(round(x)), int(round(y)))
    cv2.circle(img, point, radius, (0, 0, 255), -1)
    cv2.putText(img, label, (point[0] + radius, point[1] - radius),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

# 7. Sonucu results/{name}_gt.png olarak kaydetme
out_path = f"results/{name}_gt.png"
cv2.imwrite(out_path, img)
print(f"Kaydedildi: {out_path}")
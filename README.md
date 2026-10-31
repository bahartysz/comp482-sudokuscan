# SudokuScan

COMP 482, Proje 1. Basılı bir sudokunun fotoğrafından 9×9 dijital grid çıkarır.

| Aşama | İş |
|---|---|
| Stage 1 | Griye çevirme, küçültme, bulanıklaştırma, aydınlatma dengeleme |
| Stage 2 | Harris adayları, konturla sudoku bölgesi, en yakın aday ile köşe seçimi, TL-TR-BR-BL sıralaması |
| Stage 3 | A3, A4 ve H4 rektifikasyonu (450×450) |
| Stage 4 | Afin ve homografi karşılaştırması (ayrı scriptler, ana programın parçası değil) |
| Stage 5 | Hücre bölme, boş/dolu kararı, Tesseract ile OCR |
| Stage 6 | Satır, sütun ve kutu geçerlilik kontrolü |

## Kurulum

Python 3.12 veya üstü gerekir (`numpy` ve `scipy` sürümleri bunu ister). Python 3.13 ve Tesseract 5.5 ile denendi.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

brew install tesseract              # macOS
sudo apt install tesseract-ocr      # Ubuntu
tesseract --version                 # kurulumu test etmek için
```

## Kullanım

```bash
# Tek fotoğraf
python sudoku_scan.py --input photo.jpg --outdir out/

# Klasördeki bütün fotoğraflar
python sudoku_scan.py --batch images_dir/ --outdir results/
```

Her `<ad>` fotoğrafı için çıktı klasörüne şunlar yazılır:

| Dosya | İçerik |
|---|---|
| `<ad>_grid.txt` | 9 satır, 9 rakam, ayraç yok, boş hücre = 0 (H4 rektifikasyonundan) |
| `<ad>_result.json` | Köşeler (TL, TR, BR, BL; orijinal pikseller), A3/A4/H4 matrisleri, A3 4. köşe hatası, A4 RMS, grid, karar, süre |
| `<ad>_harris.png` | Harris yanıt haritası, bütün adaylar ve seçilen 4 köşe |
| `<ad>_rect_A3.png`, `_rect_A4.png`, `_rect_H4.png` | 450×450 düzeltilmiş görüntüler |
| `<ad>_debug.png` | H4 görüntüsü, hücre sınırları, okunan rakamlar ve çakışan hücreler |

`result.json` içindeki `verdict` alanı üç değer alır:

- `valid`: okunan rakamlar sudoku kurallarına uyuyor. Bu, okumanın doğru olduğunu garanti etmez.
- `invalid`: satır, sütun ya da kutuda tekrar eden rakam var; büyük olasılıkla OCR hatasıdır.
- `not_found`: bulmaca bulunamadı ya da bir hata oluştu. Program çökmez; `reason` alanı sebebi yazar.

## Değerlendirme

`evaluate.py`, program çıktılarını etiketlerle karşılaştırır. Köşe hatası, hücre doğruluğu, boş-hücre F1, tam eşleşme ve sağlamlığı toplamda ve C1–C5 koşullarına göre yazdırır.

```bash
python sudoku_scan.py --batch data/dev/images --outdir results/dev
python evaluate.py --results results/dev --labels data/dev/annotations

python sudoku_scan.py --batch data/personal --outdir results/personal
python evaluate.py --results results/personal --labels data/personal
```

Bu fotoğraflardaki sonuçlar:

| | Geliştirme (n=20) | Kişisel (n=12) |
|---|---|---|
| Köşe hatası | %0.64 | %0.29 |
| Hücre doğruluğu | %99.4 | %98.9 |
| Boş-hücre F1 | 0.995 | 0.996 |
| Tam eşleşme | 14/20 | 7/12 |
| Bulunan | 20/20 | 12/12 |
| Sağlamlık | 20/20 | 12/12 |
| Süre / fotoğraf | 1.3–1.8 s | 1.6–1.9 s |

## Proje yapısı

```text
sudoku_scan.py                 Ana program
evaluate.py                    Metrik hesabı
pipeline/
  stage1_preprocess.py         Griye çevirme, küçültme, bulanıklaştırma, aydınlatma dengeleme
  stage2_corners.py            Harris, bölge bulma, köşe seçimi ve sıralama
  stage3_rectify.py            A3, A4 ve H4 rektifikasyonu
  stage5_cells_ocr.py          Hücre bölme, boş/dolu kararı, OCR
  stage6_validate.py           Geçerlilik kontrolü
stage4_compare.py              Stage 4: eğiklik, A3 4. köşe hatası, A4 RMS
stage4_grid.py                 Stage 4: ızgara çizgisi sapması
stage4_ocr.py                  Stage 4: A3, A4 ve H4 üzerinde OCR doğruluğu
stage4_figure.py               Stage 4: en eğik fotoğrafın yan yana şekli
param_study.py                 Stage 2 parametre çalışması
confusion_crops.py             Yanlış okunan hücre kesitleri
margin_study.py                Stage 5 kenar payı karşılaştırması
annotate.py                    Köşe etiketleme aracı
dev_tools.py, explore_harris.py, view_*.py, test_*.py    Geliştirme ve görselleştirme araçları 
```

`data/` ve `results/` klasörleri `.gitignore` içindedir, yani depoda yer almaz. Geliştirme seti ve kişisel dataset LMS üzerinden teslim edilir; değerlendirme komutları için bu klasör gerekir.

## Kişisel dataset

Samsung S25 Ultra ile flaşsız ve 12 MP çekilmiş 12 fotoğraf, üç farklı bulmaca kaynağı. Dosyalar `s332501004_01..12.jpg`, etiketler aynı adlı JSON. Koşul dağılımı: C1: 2, C2: 5, C3: 1, C4: 2, C5: 2.

## Ayarlanabilir sabitler

| Dosya | Sabitler |
|---|---|
| `pipeline/stage2_corners.py` | `HARRIS_BLOCK`, `HARRIS_KSIZE`, `HARRIS_K`, `THR`, `NMS_FRAC`, `SEARCH_FRAC` |
| `pipeline/stage3_rectify.py` | `CELL`, `A3_CORNERS` |

## Sınırlamalar

- Fotoğrafta sudokudan büyük bir dikdörtgen (kitap, çerçeve) varsa yanlış bölge seçilebilir.
- Harris köşeleri gerçek köşeden içeri doğru birkaç piksel kayar.
- Çok eğik çekimlerde uzak taraftaki rakamlar küçük ve bulanık kalır, OCR doğruluğu düşer.
- Kâğıt kıvrımı 3×3 dönüşümle tam düzeltilemez.
- Sudoku 45°'den fazla dönmüşse sol üst köşe yanlış belirlenir.

# Banana Ripeness Classification — Classical Computer Vision

Muz fotoğraflarından olgunluk seviyesini (`unripe` / `ripe` / `overripe`) sınıflandıran sistem.
Öznitelik vektörüne kadar **yalnızca klasik görüntü işleme** (HSV segmentasyonu, renk histogramları,
kahverengi leke analizi, GLCM, LBP) kullanılır; sınıflandırma scikit-learn ile yapılır.
Ana deney: RGB, HSV ve Lab renk uzaylarının sınıflandırma başarısına etkisi.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Kullanım

```bash
python -m src.extract_features --data data/raw --out data/features/features.csv --save-masks 20
python -m src.train --features data/features/features.csv --feature-set all --out models/
python -m src.compare_color_spaces --features data/features/features.csv --data data/raw --out results/
python -m src.evaluate --model models/best_model.joblib --features data/features/features.csv
python -m src.predict --image path/to/banana.jpg
pytest tests/
```

## Proje Yapısı

- `src/` — pipeline kodu (ön işleme → segmentasyon → öznitelik → sınıflandırma)
- `data/` — veri seti (git'e eklenmez, bkz. `data/README.md`)
- `notebooks/` — keşif ve parametre ayarlama
- `results/` — figürler, örnek maskeler, metrikler
- `report/` — akademik rapor

## Sonuçlar

_Deneyler tamamlandığında eklenecek._

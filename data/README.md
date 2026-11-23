# data/

Veri dosyaları git'e eklenmez; yalnızca klasör yapısı tutulur.

- `raw/<sınıf>/` — Kaggle veri setinin orijinal görüntüleri (`unripe`, `ripe`, `overripe`).
  Veri setindeki fazladan sınıflar `src/config.py` içindeki `CLASS_MAP` ile eşlenir veya çıkarılır.
- `processed/` — opsiyonel ön işleme/maske önbelleği.
- `features/` — `features.csv` ve parlaklık testi kopyaları (`features_brightness_*.csv`).

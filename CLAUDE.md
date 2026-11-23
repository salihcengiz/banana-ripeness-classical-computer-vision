# CLAUDE.md — Banana Ripeness Classification (Classical Computer Vision)

Bu dosya, Claude Code'un bu projede nasıl çalışacağını tanımlar. Her oturumda önce bunu oku ve kurallara uy.

## 1. Proje Özeti

Muz görüntülerinden **olgunluk seviyesini** sınıflandıran bir sistem.

- **Sınıflar (ödev tanımı, 3 sınıf):** `unripe` (ham/yeşil), `ripe` (olgun/sarı), `overripe` (aşırı olgun/kahverengi benekli). Sınıf listesi `src/config.py` içinde tek yerde tutulur. Kaggle veri setinde fazladan sınıf varsa (ör. `rotten`), `config.CLASS_MAP` ile ya `overripe`'a eşlenir ya da veri setinden çıkarılır; bu karar raporda belirtilir.
- **Girdi:** Muz fotoğrafı (`.jpg`, `.png`). Veri kaynağı: Kaggle'daki hazır meyve/muz olgunluk veri setleri.
- **Çıktı:** Sınıf etiketi + güven skoru, ayrıca ara aşama görselleri (maske, leke haritası vb.).
- **Teslimatlar:** Çalışan kod, sonuç grafikleri ve metodolojiyi ayrıntılı anlatan **rapor**.

### Ödevin zorunlu içeriği
1. **HSV'de arka plan ayırma** (segmentasyon).
2. Öznitelikler: **ortalama ton (hue) değeri**, **renk histogramı**, **kahverengi leke oranı**.
3. AI kısmında bu özniteliklerle **sınıflandırma**.
4. **Derinlik deneyi (projenin ana katkısı):** **RGB, HSV ve Lab** renk uzaylarının sınıflandırma başarısına etkisinin karşılaştırılması → bkz. Bölüm 4, Aşama 5.

## 2. KESİN KURAL — Yasak / Serbest Ayrımı

Proje iki aşamadan oluşur. Sınır, **görüntünün sayısal bir öznitelik vektörüne dönüştüğü andır.**

```
[Görüntü] → Ön işleme → Segmentasyon → Öznitelik çıkarma → [Öznitelik vektörü] → Sınıflandırıcı → [Etiket]
|<------------------- SADECE KLASİK GÖRÜNTÜ İŞLEME ------------------->|<------- ML/AI SERBEST ------->|
```

### Vektörden ÖNCE kesinlikle YASAK
- CNN, YOLO, U-Net, ResNet, VGG, ViT, SAM, Mask R-CNN veya **herhangi bir sinir ağı**.
- Önceden eğitilmiş modellerden embedding/feature çıkarmak (ör. ImageNet backbone ile feature extraction).
- `torch`, `tensorflow`, `keras`, `ultralytics`, `rembg`, `transformers`, `timm` gibi kütüphanelerin görüntü işleme aşamasında kullanılması.
- Öğrenilmiş arka plan silme araçları.

### Vektörden ÖNCE serbest (kullanılması beklenen) yöntemler
- Renk uzayı dönüşümleri: RGB → HSV, Lab, YCrCb, Grayscale.
- Filtreleme: Gaussian, Median, Bilateral blur; histogram eşitleme, **CLAHE**; gray-world beyaz dengesi.
- Eşikleme: sabit eşik, **Otsu**, adaptive threshold, renk aralığı (`cv2.inRange`).
- Kümeleme ile segmentasyon: **K-means** (piksel renkleri üzerinde — bu klasik bir yöntem olarak kabul edilir), GrabCut, Watershed.
- Morfoloji: erosion, dilation, opening, closing, hole filling.
- Kenar/kontur: Canny, Sobel, `findContours`, connected components.
- Doku: **GLCM** (Haralick), **LBP**, Gabor filtreleri.
- Şekil: alan, çevre, solidity, eccentricity, Hu momentleri.

### Vektörden SONRA serbest
- scikit-learn sınıflandırıcıları: SVM, kNN, Random Forest, Logistic Regression, Gradient Boosting, MLP (sadece vektör üzerinde).
- Öznitelik ölçekleme, PCA, öznitelik seçimi, GridSearchCV, cross-validation.

> **Claude için talimat:** Bir öneri veya kod bu kuralı ihlal ediyorsa yazma; kullanıcıyı uyar. `requirements.txt` dosyasına derin öğrenme kütüphanesi ekleme. Emin olmadığın bir yöntem varsa önce sor.

## 3. Dosya Sistemi

```
banana-ripeness-classical-computer-vision/
├── CLAUDE.md                 # Bu dosya: proje kuralları ve rehberi
├── README.md                 # Kurulum, kullanım ve kısa sonuç özeti
├── requirements.txt          # Bağımlılıklar (DL kütüphanesi YOK)
├── .gitignore                # data/, results/ çıktıları, __pycache__, .venv vb.
│
├── data/                     # ⚠️ git'e eklenmez (sadece README/.gitkeep)
│   ├── raw/                  # Orijinal veri seti, sınıf başına bir klasör
│   │   ├── unripe/
│   │   ├── ripe/
│   │   └── overripe/
│   ├── processed/            # Ön işlenmiş görüntüler ve maskeler (opsiyonel önbellek)
│   └── features/
│       ├── features.csv      # Her satır = bir görüntü: dosya yolu, etiket, TÜM öznitelikler
│       │                     #   Sütun önekleri: rgb_*, hsv_*, lab_*, spot_*, glcm_*, lbp_*, shape_*
│       └── features_brightness_*.csv  # Parlaklık değiştirilmiş test kopyaları (dayanıklılık testi)
│
├── src/                      # Tüm kaynak kod
│   ├── __init__.py
│   ├── config.py             # Sabitler: sınıflar, görüntü boyutu, eşik değerleri, yollar, random seed
│   ├── io_utils.py           # Görüntü okuma/yazma, veri setini klasörlerden listeleme
│   ├── preprocess.py         # AŞAMA 1: resize, denoise (median/gaussian), beyaz dengesi, CLAHE
│   ├── segment.py            # AŞAMA 2: muzu arka plandan ayırma → ikili maske
│   │                         #   HSV'de S kanalı Otsu + hue aralığı, opsiyonel K-means/GrabCut,
│   │                         #   morfoloji, en büyük kontur, delik doldurma
│   ├── color_spaces.py       # BGR → RGB / HSV / Lab dönüşümleri ve kanal aralıkları
│   │                         #   (her uzay için kanal adları, min-max değerleri, histogram aralığı)
│   ├── features.py           # AŞAMA 3: maske içinden öznitelik vektörü çıkarma
│   │                         #   color_features(img, mask, space) → seçilen uzayda kanal
│   │                         #     istatistikleri + kanal histogramları (rgb_/hsv_/lab_ önekli)
│   │                         #   spot_features → kahverengi leke oranı, sayısı, boyutu
│   │                         #   texture/shape → GLCM, LBP, şekil (ek öznitelikler)
│   ├── extract_features.py   # Tüm veri seti için 1→2→3'ü çalıştırır, features.csv üretir
│   │                         #   (3 renk uzayının öznitelikleri aynı CSV'de, önekle ayrılmış)
│   ├── train.py              # AŞAMA 4: CSV'yi okur, birden çok sınıflandırıcıyı eğitir/karşılaştırır
│   │                         #   --feature-set rgb|hsv|lab|hsv+lab|all ile öznitelik grubu seçilir
│   ├── compare_color_spaces.py  # AŞAMA 5: RGB vs HSV vs Lab deneyi (ana derinlik deneyi)
│   │                         #   aynı split/seed/modellerle her uzayı eğitir, tablo + grafik üretir
│   ├── evaluate.py           # Confusion matrix, classification report, yanlış sınıflanan örnekler
│   ├── predict.py            # Tek görüntü için uçtan uca tahmin (CLI)
│   └── visualize.py          # Rapor için ara aşama figürleri (pipeline adım adım)
│
├── notebooks/                # Keşif ve deneme (rapora girecek grafikleri üretmek için)
│   ├── 01_color_space_exploration.ipynb   # Kanalların sınıfları ne kadar ayırdığı
│   ├── 02_segmentation_tuning.ipynb       # Eşik/morfoloji parametre denemeleri
│   ├── 03_feature_analysis.ipynb          # Öznitelik dağılımları, korelasyon, önem
│   └── 04_color_space_results.ipynb       # Aşama 5 sonuçlarının yorumlanması, rapor grafikleri
│
├── models/                   # Eğitilmiş model (.joblib) + scaler — git'e eklenmez
│
├── results/                  # Üretilen çıktılar
│   ├── figures/              # Pipeline görselleri, histogramlar, confusion matrix
│   ├── masks/                # Örnek segmentasyon maskeleri (kalite kontrolü için)
│   └── metrics/              # model_comparison.csv, classification_report.txt,
│                             # color_space_comparison.csv, color_space_robustness.csv
│
├── report/                   # Akademik rapor
│   ├── report.md             # (veya report.tex / .docx) — ana rapor
│   └── figures/              # Rapora konacak seçilmiş figürler
│
└── tests/                    # Basit birim testleri
    ├── test_segment.py       # Maske boş değil mi, doğru boyutta mı
    └── test_features.py      # Vektör uzunluğu sabit mi, NaN yok mu
```

### Veri akışı
```
data/raw/<sınıf>/*.jpg
   └─(extract_features.py)─→ preprocess → segment → features
        └─→ data/features/features.csv  (+ results/masks/ örnekleri)
             ├─(train.py)─→ models/best_model.joblib + results/metrics/
             ├─(compare_color_spaces.py)─→ results/metrics/color_space_comparison.csv
             │                             + results/figures/color_space_*.png
             └─(evaluate.py / visualize.py)─→ results/figures/ → report/
```

## 4. Pipeline Ayrıntıları (Varsayılan Tasarım)

### Aşama 1 — Ön İşleme (`preprocess.py`)
1. Sabit boyuta getirme (ör. uzun kenar 512 px, en-boy oranı korunur).
2. Gürültü azaltma: Median blur (k=5) — tuz-biber gürültüsünde kenarları korur.
3. Beyaz dengesi: Gray-world varsayımı — farklı ışık koşullarındaki renk kaymasını azaltır.
4. Kontrast: Lab uzayında yalnızca L kanalına CLAHE — renk bilgisini bozmadan aydınlatmayı dengeler.

### Aşama 2 — Segmentasyon (`segment.py`)
1. HSV'ye dönüştür; muz piksellerinin doygunluğu (S) arka plandan yüksektir → S kanalına **Otsu** eşikleme.
2. Gerekirse hue aralığı ile birleştir (yeşil–sarı–kahverengi aralığı).
3. Morfolojik **opening** (gürültü) + **closing** (boşluk) + delik doldurma (kahverengi lekeler maskeden düşmesin diye kritik).
4. Connected components → **en büyük bileşen** = muz.
5. Alternatif yöntemler (karşılaştırma için): K-means (k=2/3) Lab üzerinde, GrabCut (Otsu maskesini başlangıç olarak kullanarak).

> Segmentasyon **her deneyde HSV'de ve aynı parametrelerle** yapılır. Böylece Aşama 5'teki renk uzayı karşılaştırmasında maskeler sabit kalır; fark yalnızca öznitelik çıkarılan renk uzayından gelir.

### Aşama 3 — Öznitelik Çıkarma (`features.py`) — SADECE maske içi pikseller

**A) Renk uzayına bağlı öznitelikler** — her uzay için **aynı şablonla** çıkarılır (adil karşılaştırma için):

| Öznitelik | RGB (`rgb_*`) | HSV (`hsv_*`) | Lab (`lab_*`) |
|---|---|---|---|
| Kanal ortalaması | R, G, B | H (**ortalama ton**), S, V | L, a*, b* |
| Kanal std + medyan | R, G, B | H, S, V | L, a*, b* |
| Kanal histogramı (normalize, kanal başı 16 bin) | 3×16 | 3×16 | 3×16 |

Her uzayda vektör boyutu eşittir (3 kanal × (3 istatistik + 16 bin) = **57**). Böylece başarı farkı boyuttan değil, renk temsilinden kaynaklanır.

Dikkat: Hue döngüsel bir değişkendir (0 ve 179 komşudur). Ortalama ton için **dairesel ortalama** kullan (`atan2(mean(sin h), mean(cos h))`). Muzda kırmızıya yakın ton nadir olduğu için fark küçük olabilir ama raporda belirtilmeli.

**B) Renk uzayından bağımsız (sabit) öznitelikler:**

| Grup | Öznitelikler | Neden |
|---|---|---|
| **Kahverengi leke analizi** | Leke alanı oranı (leke pikseli / muz pikseli), leke sayısı, ortalama leke boyutu | Olgun vs aşırı olgun ayrımının en güçlü göstergesi |
| Renk oranları | Yeşil piksel oranı, sarı piksel oranı | Ham vs olgun için yorumlanabilir gösterge |
| Doku (GLCM) | Contrast, homogeneity, energy, correlation, dissimilarity | Kabuk yüzeyindeki bozulma |
| Doku (LBP) | Uniform LBP histogramı (P=8, R=1 → 10 bin) | Yerel doku desenleri |
| Şekil | Solidity, eccentricity, alan oranı | Deformasyon |

Kahverengi leke tanımı (HSV, OpenCV ölçeği): yaklaşık H ∈ [5, 30], düşük–orta V (ör. V < 120) **veya** çok düşük V (koyu leke). Eşikler `config.py`'de tutulur ve notebook 02'de veriyle ayarlanır.

Vektör uzunluğu **her görüntü için sabit** olmalı; NaN çıkmamalı (boş maske durumunu ele al).

### Aşama 4 — Sınıflandırma (`train.py`)
1. **Baseline:** Kural tabanlı sınıflandırıcı (ör. yeşil oranı ve leke oranı eşikleri) — ML olmadan ne kadar gidilebildiğini gösterir.
2. **ML modelleri:** SVM (RBF), kNN, Random Forest, Logistic Regression.
3. `StandardScaler` + `Pipeline`, stratified train/test ayrımı (%80/%20) ve 5-fold cross-validation, GridSearchCV.
4. Random Forest ile **öznitelik önem analizi** → hangi öznitelik grubunun işe yaradığı rapora girer.
5. **Ablation:** renk / renk + leke / renk + leke + doku / tümü karşılaştırması.

### Aşama 5 — Renk Uzayı Karşılaştırma Deneyi (`compare_color_spaces.py`) — ANA DERİNLİK DENEYİ

**Soru:** Aynı segmentasyon, aynı öznitelik şablonu ve aynı sınıflandırıcılar kullanıldığında, öznitelikleri RGB, HSV veya Lab'dan çıkarmak sınıflandırma başarısını nasıl etkiler?

**Deney protokolü (adil karşılaştırma için hepsi sabit):**
- Aynı maskeler (HSV segmentasyonu), aynı train/test ayrımı, aynı `SEED`.
- Aynı öznitelik şablonu ve aynı vektör boyutu (Aşama 3-A).
- Aynı modeller: SVM (RBF), kNN, Random Forest, Logistic Regression; aynı hiperparametre ızgarası.
- Metrikler: accuracy, F1-macro; **5-fold stratified CV ortalama ± std** + ayrı test seti skoru.

**Deney konfigürasyonları:**

| # | Öznitelik seti | Amaç |
|---|---|---|
| E1 | Sadece `rgb_*` | Referans (ham renk temsili) |
| E2 | Sadece `hsv_*` | Ton/doygunluk/parlaklık ayrımının etkisi |
| E3 | Sadece `lab_*` | Algısal olarak düzgün uzayın etkisi |
| E4 | `hsv_*` + `lab_*` | Uzayların birleşmesi ek bilgi katıyor mu? |
| E5 | E1/E2/E3 + sabit öznitelikler (leke, doku, şekil) | Renk uzayı farkı, ek öznitelikler varken de sürüyor mu? |

**Ek analizler (raporu güçlendirir):**
1. **Aydınlatma dayanıklılığı testi:** Test görüntülerinin parlaklığı yapay olarak değiştirilir (V/L ölçekleme veya gamma: 0.6, 0.8, 1.2, 1.4). Model orijinal veride eğitilir, değiştirilmiş veride test edilir. Beklenti: RGB'de başarı daha fazla düşer, çünkü RGB'de renk ve parlaklık iç içedir; HSV'de H, Lab'da a*/b* parlaklıktan büyük ölçüde ayrıdır.
2. **Tek kanal analizi:** Her kanal (R, G, B, H, S, V, L, a*, b*) tek başına kullanılarak hangi kanalın en ayırt edici olduğu gösterilir.
3. **Sınıf ayrışma görselleştirmesi:** Her uzay için kanal ortalamalarının sınıflara göre kutu grafikleri (boxplot) ve 2B dağılım grafikleri (ör. H–S, a*–b*, R–G).
4. **İstatistiksel karşılaştırma (opsiyonel):** CV skorları üzerinde eşleştirilmiş t-testi veya Wilcoxon testi; farkın anlamlı olup olmadığı.

**Çıktılar:**
- `results/metrics/color_space_comparison.csv` → satır = (uzay, model), sütunlar = CV ortalama, CV std, test accuracy, test F1-macro.
- `results/metrics/color_space_robustness.csv` → parlaklık seviyesine göre başarı.
- `results/figures/color_space_bar.png` (gruplu çubuk grafik, hata çubuklarıyla), `color_space_robustness.png` (çizgi grafik), `color_space_confusion_*.png`, `channel_boxplots_*.png`.

> **Claude için talimat:** Sonucu önceden varsayma. Beklenti (HSV/Lab > RGB) bir hipotezdir; veri tersini gösterirse olduğu gibi raporla ve nedenini tartış (ör. kontrollü ışıkta çekilmiş veri setinde RGB de iyi sonuç verebilir).

## 5. Komutlar

```bash
# Kurulum
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Öznitelik çıkarma (tüm veri seti)
python -m src.extract_features --data data/raw --out data/features/features.csv --save-masks 20

# Eğitim + model karşılaştırma (öznitelik seti seçilebilir)
python -m src.train --features data/features/features.csv --feature-set all --out models/

# Renk uzayı karşılaştırma deneyi (E1–E5 + dayanıklılık testi)
python -m src.compare_color_spaces --features data/features/features.csv --data data/raw --out results/

# Değerlendirme ve figürler
python -m src.evaluate --model models/best_model.joblib --features data/features/features.csv
python -m src.visualize --image data/raw/ripe/ornek.jpg --out results/figures/

# Tek görüntü tahmini
python -m src.predict --image path/to/banana.jpg

# Testler
pytest tests/
```

`requirements.txt` içeriği: `opencv-python`, `numpy`, `scikit-image`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `joblib`, `tqdm`, `pytest`, `jupyter`.

## 6. Kodlama Kuralları

- Python 3.10+, her fonksiyonda type hint ve kısa docstring (docstring'de **hangi yöntemin neden** kullanıldığı yazsın — rapor yazarken işe yarar).
- Tüm sabit sayılar (eşikler, kernel boyutları, bin sayıları) `config.py` içinde; kod içine gömme.
- OpenCV görüntüleri BGR okur — dönüşümlerde buna dikkat et; içeride tutarlı olarak **BGR** kullan, sadece matplotlib ile gösterirken RGB'ye çevir.
- OpenCV'de Hue 0–179 aralığındadır (0–360 değil); eşikleri buna göre yaz.
- `random_state` / seed her yerde `config.SEED` ile sabit (tekrarlanabilirlik).
- Her aşama fonksiyonu saf (pure) olsun: görüntü al → sonuç döndür; dosya yazma işi `extract_features.py`/`visualize.py`'de.
- Büyük dosyaları (veri, model) git'e ekleme.

## 7. Rapor Gereksinimleri (ÖNEMLİ)

Raporda **metodoloji ayrıntılı** anlatılmalı. Her adım için: *ne yapıldı, neden bu yöntem seçildi, parametreler ne, alternatiflere göre etkisi ne.*

Önerilen bölümler:
1. **Giriş** — Problem, motivasyon, derin öğrenme yasağının çerçevesi.
2. **Veri Seti** — Kaynak, sınıf dağılımı (tablo/bar grafik), örnek görüntüler, train/test ayrımı.
3. **Metodoloji**
   - 3.1 Ön işleme — her adımın öncesi/sonrası görseli.
   - 3.2 Renk uzayları — RGB, HSV ve Lab'ın tanımı, dönüşüm formülleri, her birinin renk ve parlaklığı nasıl temsil ettiği; OpenCV değer aralıkları (H: 0–179, Lab 8-bit ölçekleme).
   - 3.3 Segmentasyon — neden HSV'de yapıldığı, Otsu eşiği, morfoloji adımları, başarılı ve başarısız maske örnekleri.
   - 3.4 Öznitelik çıkarma — ortalama ton (dairesel ortalama), renk histogramı, kahverengi leke oranı ve ek özniteliklerin formülü/tanımı ve gerekçesi (GLCM ve LBP'nin matematiksel tanımı dahil).
   - 3.5 Öznitelik vektörü — her uzay için aynı şablon, toplam boyut, normalizasyon.
   - 3.6 Sınıflandırma — kullanılan modeller, hiperparametre arama, cross-validation.
   - 3.7 Renk uzayı karşılaştırma deney tasarımı — sabit tutulanlar, E1–E5 konfigürasyonları, dayanıklılık testi.
4. **Deneyler ve Sonuçlar**
   - 4.1 Model karşılaştırma tablosu, confusion matrix, öznitelik önem grafiği.
   - 4.2 **Renk uzayı karşılaştırması** — sonuç tablosu, gruplu çubuk grafik, kanal bazlı boxplot'lar, tek kanal analizi.
   - 4.3 **Aydınlatma dayanıklılığı** — parlaklık değişimine göre başarı grafiği.
   - 4.4 Ablation sonuçları.
   - 4.5 **Tartışma** — hangi uzay neden daha iyi/kötü; renk–parlaklık ayrımının rolü; bulguların renk uzayı teorisiyle tutarlılığı.
5. **Hata Analizi** — Yanlış sınıflanan örnekler ve nedenleri (ışık, arka plan, sınıf sınırı belirsizliği).
6. **Sonuç ve Gelecek Çalışmalar.**
7. **Kaynaklar** — Otsu (1979), Haralick (1973) GLCM, Ojala (2002) LBP, Zuiderveld (1994) CLAHE vb.

> **Claude için talimat:** Yeni bir aşama veya parametre eklediğinde, raporun ilgili metodoloji bölümüne girecek 2–4 cümlelik gerekçeyi de öner ve gerekiyorsa `visualize.py` ile ilgili figürü üret.

## 8. Önerilen Çalışma Sırası

1. [ ] Repo iskeleti, `requirements.txt`, `.gitignore`, `config.py`
2. [ ] Veri setini `data/raw/<sınıf>/` yapısına yerleştir, sınıf dağılımını kontrol et
3. [ ] `notebooks/01` — renk uzayı keşfi
4. [ ] `preprocess.py` + `segment.py` → 20–30 görüntüde maskeleri gözle kontrol et
5. [ ] `features.py` + `extract_features.py` → `features.csv`
6. [ ] Kural tabanlı baseline
7. [ ] `train.py` + `evaluate.py` → model karşılaştırması
8. [ ] `compare_color_spaces.py` → RGB vs HSV vs Lab (E1–E5) + aydınlatma dayanıklılığı testi
9. [ ] Ablation + öznitelik önemi
10. [ ] `visualize.py` ile rapor figürleri
11. [ ] Rapor yazımı

## 9. Sık Karşılaşılan Sorunlar

- **Kahverengi lekeler maskeden düşüyor** → closing kernel'ını büyüt + `fill holes` uygula; leke analizini maske içinde yap.
- **Beyaz/gri arka plan sarı muzla karışıyor** → S kanalı eşiği yetmezse Lab b* kanalını da kullan.
- **Gölgeler muz sanılıyor** → V kanalında çok düşük değerleri ve düşük doygunluğu dışla.
- **Sınıflar dengesiz** → `class_weight="balanced"` ve stratified split; raporda F1-macro raporla.
- **Veri setinde 3'ten fazla sınıf var** → `config.CLASS_MAP` ile eşle veya çıkar; kararı raporun "Veri Seti" bölümüne yaz.
- **Lab değerleri beklenenden farklı** → OpenCV 8-bit Lab'da L 0–255'e ölçeklenir, a* ve b* 128 merkezlidir; raporda gerçek aralıklara çevirerek anlat.
- **Birden fazla muz (salkım)** → en büyük bileşen yerine alanı belli eşiğin üstündeki tüm bileşenleri birleştir.

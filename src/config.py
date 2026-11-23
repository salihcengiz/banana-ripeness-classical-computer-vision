"""Proje genelindeki tüm sabitler.

Eşikler, kernel boyutları, bin sayıları, yollar ve seed yalnızca burada tutulur;
diğer modüller bu değerleri buradan okur (kod içine sayı gömülmez).

Notlar:
- OpenCV görüntüleri BGR okur; içeride BGR kullanılır.
- OpenCV'de Hue 0–179 aralığındadır; tüm hue eşikleri bu ölçektedir.
- OpenCV 8-bit Lab: L 0–255'e ölçeklenir, a* ve b* 128 merkezlidir.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FEATURES_DIR: Path = DATA_DIR / "features"
FEATURES_CSV: Path = FEATURES_DIR / "features.csv"
MODELS_DIR: Path = ROOT_DIR / "models"
RESULTS_DIR: Path = ROOT_DIR / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
MASKS_DIR: Path = RESULTS_DIR / "masks"
METRICS_DIR: Path = RESULTS_DIR / "metrics"

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

# ---------------------------------------------------------------------------
# Tekrarlanabilirlik
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Sınıflar
# ---------------------------------------------------------------------------
CLASSES: tuple[str, ...] = ("unripe", "ripe", "overripe")

# Veri setindeki klasör adı → proje sınıfı. None = veri setinden çıkar.
# Kaggle veri setine göre güncellenir; karar raporun "Veri Seti" bölümüne yazılır.
CLASS_MAP: dict[str, str | None] = {
    "unripe": "unripe",
    "ripe": "ripe",
    "overripe": "overripe",
    "rotten": None,  # TODO: veri seti incelendikten sonra "overripe" veya None
}

# ---------------------------------------------------------------------------
# Aşama 1 — Ön işleme
# ---------------------------------------------------------------------------
RESIZE_LONG_SIDE: int = 512          # uzun kenar (px), en-boy oranı korunur
MEDIAN_KSIZE: int = 5                # median blur kernel (tek sayı)
CLAHE_CLIP_LIMIT: float = 2.0        # yalnızca Lab L kanalına uygulanır
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)
USE_GRAY_WORLD: bool = True
USE_CLAHE: bool = True

# ---------------------------------------------------------------------------
# Aşama 2 — Segmentasyon (her deneyde HSV'de, aynı parametrelerle)
# ---------------------------------------------------------------------------
# Muzun yeşil–sarı–kahverengi hue aralığı (OpenCV 0–179)
BANANA_HUE_RANGE: tuple[int, int] = (5, 90)
USE_HUE_RANGE: bool = True
SHADOW_V_MAX: int = 40               # bunun altındaki V gölge sayılır
SHADOW_S_MAX: int = 40               # ...ve doygunluk da düşükse dışlanır
MORPH_OPEN_KSIZE: int = 5
MORPH_CLOSE_KSIZE: int = 15          # büyük: kahverengi lekeler maskeden düşmesin
MIN_COMPONENT_AREA_RATIO: float = 0.01  # salkım modu için alt alan eşiği
KMEANS_K: int = 2                    # alternatif segmentasyon (Lab üzerinde)
KMEANS_ATTEMPTS: int = 3
GRABCUT_ITERS: int = 5

# ---------------------------------------------------------------------------
# Aşama 3 — Öznitelikler
# ---------------------------------------------------------------------------
COLOR_SPACES: tuple[str, ...] = ("rgb", "hsv", "lab")
HIST_BINS: int = 16                  # kanal başına; 3 × (3 istatistik + 16) = 57

# Kahverengi leke (HSV, OpenCV ölçeği) — notebook 02'de veriyle ayarlanır
SPOT_HUE_RANGE: tuple[int, int] = (5, 30)
SPOT_V_MAX: int = 120                # düşük–orta parlaklık
SPOT_DARK_V_MAX: int = 60            # çok koyu leke (hue'dan bağımsız)
SPOT_MIN_AREA: int = 10              # px; bundan küçük bileşenler gürültü

# Renk oranları (HSV)
GREEN_HUE_RANGE: tuple[int, int] = (35, 85)
YELLOW_HUE_RANGE: tuple[int, int] = (20, 35)
COLOR_RATIO_S_MIN: int = 50          # düşük doygunluklu pikseller sayılmaz

# GLCM
GLCM_DISTANCES: tuple[int, ...] = (1, 3)
GLCM_ANGLES_DEG: tuple[int, ...] = (0, 45, 90, 135)
GLCM_LEVELS: int = 32
GLCM_PROPS: tuple[str, ...] = (
    "contrast", "homogeneity", "energy", "correlation", "dissimilarity",
)

# LBP (uniform, P=8, R=1 → P+2 = 10 bin)
LBP_P: int = 8
LBP_R: int = 1
LBP_METHOD: str = "uniform"

# ---------------------------------------------------------------------------
# Aşama 4 — Sınıflandırma
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
FEATURE_PREFIXES: dict[str, tuple[str, ...]] = {
    "rgb": ("rgb_",),
    "hsv": ("hsv_",),
    "lab": ("lab_",),
    "hsv+lab": ("hsv_", "lab_"),
    "fixed": ("spot_", "ratio_", "glcm_", "lbp_", "shape_"),
}

# ---------------------------------------------------------------------------
# Aşama 5 — Aydınlatma dayanıklılığı testi
# ---------------------------------------------------------------------------
BRIGHTNESS_FACTORS: tuple[float, ...] = (0.6, 0.8, 1.2, 1.4)

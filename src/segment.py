"""AŞAMA 2 — Muzu arka plandan ayırma (ikili maske).

Varsayılan yöntem HSV üzerindedir ve tüm deneylerde aynı parametrelerle çalışır:
S kanalına Otsu + hue aralığı, gölge dışlama, opening/closing, delik doldurma,
en büyük bağlı bileşen. Karşılaştırma için K-means (Lab) ve GrabCut alternatifleri.
"""

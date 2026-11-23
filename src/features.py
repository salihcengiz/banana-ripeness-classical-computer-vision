"""AŞAMA 3 — Maske içi piksellerden öznitelik vektörü çıkarma.

- ``color_features(img, mask, space)``: kanal ortalaması (H için dairesel ortalama),
  std, medyan + kanal başına ``HIST_BINS`` binlik normalize histogram → 57 boyut.
- ``spot_features``: kahverengi leke oranı, sayısı, ortalama boyutu.
- Renk oranları, GLCM, LBP ve şekil öznitelikleri.

Vektör uzunluğu her görüntüde sabittir; boş maske NaN üretmez.
"""

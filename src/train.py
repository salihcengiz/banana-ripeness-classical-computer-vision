"""AŞAMA 4 — Sınıflandırma.

CSV okunur, ``--feature-set`` (rgb|hsv|lab|hsv+lab|all) ile öznitelik grubu seçilir.
Kural tabanlı baseline + SVM, kNN, RF, LR; StandardScaler + Pipeline, stratified
split, 5-fold CV ve GridSearchCV.
"""

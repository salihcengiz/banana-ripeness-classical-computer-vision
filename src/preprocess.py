"""AŞAMA 1 — Ön işleme.

Adımlar (parametreler ``config.py``):
1. Uzun kenarı ``RESIZE_LONG_SIDE`` olacak şekilde yeniden boyutlandırma (en-boy korunur).
2. Median blur — tuz-biber gürültüsünü kenarları koruyarak azaltır.
3. Gray-world beyaz dengesi — ışık kaynağından gelen renk kaymasını azaltır.
4. Lab L kanalına CLAHE — renk bilgisini bozmadan kontrastı dengeler.
"""

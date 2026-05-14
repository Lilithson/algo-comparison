"""
generators.py -- Tekrarlanabilir sentetik graf ureteceleri.

Deneylerde dugum sayisini ve kenar yogunlugunu TAM olarak kontrol
edebilmek icin sentetik graflar uretiyoruz. Tum ureteceler bir `seed`
parametresi alir; ayni seed ayni grafi verir, boylece olcumler
tekrarlanabilir.

Uretilen graf turleri ve neden secildikleri:

  random_sparse : Bagli, seyrek rastgele graf (m ~= c * n). Cogu gercek
                  agin temel modeli; "tipik" durum.
  random_dense  : Bagli, yogun rastgele graf (m ~= p * n^2). Komsuluk
                  matrisi tabanli yaklasimlarin one ciktigi durum.
  grid_2d       : 2 boyutlu izgara. Yol aglarina benzer; buyuk capli
                  (uzun en-kisa-yollu) ve dusuk dereceli yapidir.
  scale_free    : Tercihli baglanma ile uretilen olceksiz graf. Birkac
                  yuksek dereceli "hub" icerir (sosyal ag benzeri).
  random_dag    : Yonlu cevrimsiz graf. Negatif agirliklara izin verilse
                  bile negatif CEVRIM olusamaz; Bellman-Ford'un Dijkstra'ya
                  ustunlugunu gostermek icin kullanilir.

Tum yonsuz ureteceler bagli graf garantisi verir: once rastgele bir
omurga (spanning tree) kurulur, sonra uzerine fazladan kenar eklenir.
"""
from __future__ import annotations

import random

from config import WEIGHT_MIN, WEIGHT_MAX
from graph import Graph


# -------------------------------------------------------------------------- #
# Yardimci: bagli graf garantisi icin rastgele omurga agaci                   #
# -------------------------------------------------------------------------- #
def _spanning_tree_edges(n, rng, wmin, wmax):
    """
    n dugumu birbirine baglayan rastgele bir agac uretir (n-1 kenar).
    Dugum i (karistirilmis sirada) kendisinden onceki rastgele bir
    dugume baglanir -> her zaman bagli, cevrimsiz.
    """
    order = list(range(n))
    rng.shuffle(order)
    edges = []
    for i in range(1, n):
        parent = order[rng.randrange(i)]
        child = order[i]
        edges.append((parent, child, rng.randint(wmin, wmax)))
    return edges


# -------------------------------------------------------------------------- #
# 1) Seyrek rastgele graf                                                     #
# -------------------------------------------------------------------------- #
def random_sparse(n, avg_degree=4, seed=0, wmin=WEIGHT_MIN, wmax=WEIGHT_MAX):
    """
    Bagli, yonsuz, seyrek rastgele graf.

    avg_degree : hedeflenen ortalama derece (m ~= avg_degree * n).
                 Graf yonsuz kuruldugu icin avg_degree * n / 2 yonsuz
                 kenar eklenir; Graph sinifi bunlari cift yone acar.
    """
    rng = random.Random(seed)
    target_undirected = max(n - 1, int(avg_degree * n / 2))

    edges = _spanning_tree_edges(n, rng, wmin, wmax)
    seen = {(min(u, v), max(u, v)) for (u, v, _) in edges}

    attempts = 0
    max_attempts = target_undirected * 20
    while len(edges) < target_undirected and attempts < max_attempts:
        attempts += 1
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        edges.append((u, v, rng.randint(wmin, wmax)))

    return Graph(n, edges, directed=False, name=f"seyrek-rastgele(n={n})")


# -------------------------------------------------------------------------- #
# 2) Yogun rastgele graf                                                      #
# -------------------------------------------------------------------------- #
def random_dense(n, edge_frac=0.3, seed=0, wmin=WEIGHT_MIN, wmax=WEIGHT_MAX):
    """
    Bagli, yonsuz, yogun rastgele graf.

    edge_frac : her yonsuz {u, v} ciftinin grafa girme olasiligi.
                Sonuc yogunlugu yaklasik edge_frac olur.
    """
    rng = random.Random(seed)

    edges = _spanning_tree_edges(n, rng, wmin, wmax)
    seen = {(min(u, v), max(u, v)) for (u, v, _) in edges}

    for u in range(n):
        for v in range(u + 1, n):
            key = (u, v)
            if key in seen:
                continue
            if rng.random() < edge_frac:
                seen.add(key)
                edges.append((u, v, rng.randint(wmin, wmax)))

    return Graph(n, edges, directed=False,
                 name=f"yogun-rastgele(n={n},p={edge_frac})")


# -------------------------------------------------------------------------- #
# 3) 2 boyutlu izgara grafi                                                   #
# -------------------------------------------------------------------------- #
def grid_2d(rows, cols, seed=0, wmin=WEIGHT_MIN, wmax=WEIGHT_MAX):
    """
    rows x cols 2B izgara. Her hucre sagindaki ve altindaki hucreye
    baglanir (yonsuz). Yol aglarina benzer: dusuk derece, buyuk cap.
    Dugum (i, j) -> indeks i * cols + j.
    """
    rng = random.Random(seed)
    n = rows * cols
    edges = []
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            if j + 1 < cols:  # saga
                edges.append((node, node + 1, rng.randint(wmin, wmax)))
            if i + 1 < rows:  # asagi
                edges.append((node, node + cols, rng.randint(wmin, wmax)))
    return Graph(n, edges, directed=False,
                 name=f"izgara({rows}x{cols})")


def grid_for_size(n, seed=0, wmin=WEIGHT_MIN, wmax=WEIGHT_MAX):
    """n'e en yakin (kare benzeri) izgara grafini uretir."""
    side = max(2, int(round(n ** 0.5)))
    return grid_2d(side, side, seed=seed, wmin=wmin, wmax=wmax)


# -------------------------------------------------------------------------- #
# 4) Olceksiz graf (Barabasi-Albert tercihli baglanma)                        #
# -------------------------------------------------------------------------- #
def scale_free(n, m_attach=2, seed=0, wmin=WEIGHT_MIN, wmax=WEIGHT_MAX):
    """
    Barabasi-Albert modeliyle olceksiz, yonsuz graf.

    m_attach : her yeni dugumun mevcut dugumlere actigi kenar sayisi.
               Yeni dugum, dereceyle orantili olasilikla baglanir;
               sonucta birkac yuksek dereceli "hub" olusur.
    """
    rng = random.Random(seed)
    m_attach = max(1, min(m_attach, n - 1))

    # Baslangic: m_attach + 1 dugumlu tam graf
    edges = []
    repeated = []  # tercihli secim icin: her dugum derecesi kadar tekrar eder
    init = m_attach + 1
    for u in range(init):
        for v in range(u + 1, init):
            edges.append((u, v, rng.randint(wmin, wmax)))
            repeated.append(u)
            repeated.append(v)

    # Kalan dugumleri tek tek ekle
    for new in range(init, n):
        targets = set()
        guard = 0
        while len(targets) < m_attach and guard < 1000:
            guard += 1
            targets.add(repeated[rng.randrange(len(repeated))])
        for t in targets:
            edges.append((new, t, rng.randint(wmin, wmax)))
            repeated.append(new)
            repeated.append(t)

    return Graph(n, edges, directed=False, name=f"olceksiz(n={n})")


# -------------------------------------------------------------------------- #
# 5) Yonlu cevrimsiz graf (negatif agirlik testleri icin)                     #
# -------------------------------------------------------------------------- #
def random_dag(n, avg_degree=4, allow_negative=False, neg_frac=0.25,
               seed=0, wmin=WEIGHT_MIN, wmax=WEIGHT_MAX,
               neg_min=-40, neg_max=80):
    """
    Yonlu cevrimsiz graf (DAG). Kenarlar yalnizca kucuk indeksten buyuk
    indekse gider; bu nedenle CEVRIM olusamaz.

    Her v >= 1 dugumu icin [0, v) araligindan rastgele bir gelen kenar
    eklenir -> her dugum 0'dan erisilebilir. Sonra fazladan ileri kenar
    eklenir.

    allow_negative=True ise kenarlarin neg_frac kadari negatif agirlik
    alir. DAG'da negatif cevrim olamayacagi icin en-kisa-yol hala iyi
    tanimlidir; ancak Dijkstra negatif kenarlarda yanlis sonuc verebilir,
    Bellman-Ford ise dogru sonucu uretir.
    """
    rng = random.Random(seed)

    def pick_weight():
        if allow_negative and rng.random() < neg_frac:
            return rng.randint(neg_min, neg_max)
        return rng.randint(wmin, wmax)

    edges = []
    seen = set()
    # Her dugumun 0'dan erisilebilir olmasini garantile
    for v in range(1, n):
        u = rng.randrange(v)
        edges.append((u, v, pick_weight()))
        seen.add((u, v))

    # Fazladan ileri kenarlar
    target = max(n - 1, int(avg_degree * n / 2))
    attempts = 0
    while len(edges) < target and attempts < target * 30:
        attempts += 1
        u = rng.randrange(n - 1)
        v = rng.randrange(u + 1, n)
        if (u, v) in seen:
            continue
        seen.add((u, v))
        edges.append((u, v, pick_weight()))

    suffix = "negatif" if allow_negative else "pozitif"
    return Graph(n, edges, directed=True, name=f"DAG-{suffix}(n={n})")


# -------------------------------------------------------------------------- #
# Dispatcher -- benchmark'in tek noktadan graf istemesi icin                  #
# -------------------------------------------------------------------------- #
def build(kind, size, seed=0, **kwargs):
    """
    `kind` adina gore graf uretir. Desteklenen turler:
      "sparse", "dense", "grid", "scale_free", "dag"
    """
    if kind == "sparse":
        return random_sparse(size, seed=seed, **kwargs)
    if kind == "dense":
        return random_dense(size, seed=seed, **kwargs)
    if kind == "grid":
        return grid_for_size(size, seed=seed, **kwargs)
    if kind == "scale_free":
        return scale_free(size, seed=seed, **kwargs)
    if kind == "dag":
        return random_dag(size, seed=seed, **kwargs)
    raise ValueError(f"bilinmeyen graf turu: {kind!r}")

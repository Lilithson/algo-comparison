"""
graph.py -- Agirlikli graf veri yapisi ve UC temel gosterim bicimi.

Projedeki butun algoritmalar ayni `Graph` nesnesi uzerinde calisir; boylece
karsilastirma adil olur. Bir graf bir kez kurulur ve uc farkli bicimde
talep edilebilir:

  1. Kenar listesi (edge list)         -- O(m) bellek.  Bellman-Ford'un dogal
     girdisi. Kenarlar uzerinde tek tek gezmek kolaydir.
  2. Komsuluk listesi (adjacency list) -- O(n + m) bellek. Seyrek graflarda
     verimlidir; Dijkstra (yigin) bu bicimi kullanir.
  3. Komsuluk matrisi (adjacency matrix) -- O(n^2) bellek. Iki dugum arasinda
     kenar olup olmadigi O(1) sorgulanir; yogun graflar ve Floyd-Warshall icin
     uygundur, ancak buyuk seyrek graflarda bellek acisindan pahalidir.

Gosterimler ilk talep edildiginde uretilir ve onbellege alinir (lazy cache),
boylece bir algoritmanin olcumune yalnizca kendi kullandigi gosterimin
maliyeti yansir.

Dugumler 0..n-1 araliginda tam sayilarla indekslenir.
"""
from __future__ import annotations

import sys
from config import INF


def deep_getsizeof(obj, _seen=None):
    """
    Bir Python nesnesinin ic ice yapilariyla birlikte yaklasik bellek
    ayak izini bayt cinsinden hesaplar. `sys.getsizeof` tek basina
    yalnizca ust kabugu olcer; bu fonksiyon liste/sozluk/demet/kume
    icindeki ogeleri de toplar. Bellek karsilastirmasi icin kullanilir.
    """
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += deep_getsizeof(item, _seen)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_getsizeof(k, _seen) + deep_getsizeof(v, _seen)
    return size


class Graph:
    """
    Yonlu, agirlikli graf. Yonsuz bir graf istenirse her kenar her iki
    yonde de eklenir (directed=False).
    """

    def __init__(self, n, edges, directed=True, name="graf"):
        """
        Parametreler
        ------------
        n        : dugum sayisi (dugumler 0..n-1).
        edges    : (u, v, w) ucludleri. w kenar agirligi.
        directed : True ise verilen kenarlar oldugu gibi; False ise her
                   kenar (v, u, w) olarak da eklenir.
        name     : raporlama icin okunabilir ad.
        """
        self.n = int(n)
        self.directed = directed
        self.name = name

        # Kenarlari normalize ederek tek bir kanonik listede sakla.
        norm = []
        for (u, v, w) in edges:
            u, v, w = int(u), int(v), float(w)
            norm.append((u, v, w))
            if not directed:
                norm.append((v, u, w))
        self._edges = norm

        self.m = len(self._edges)
        self.has_negative = any(w < 0.0 for (_, _, w) in self._edges)

        # Gosterim onbellekleri (lazy)
        self._adj_list = None
        self._adj_matrix = None

    # ---------------------------------------------------------------- #
    # Gosterim 1: kenar listesi                                        #
    # ---------------------------------------------------------------- #
    def edge_list(self):
        """(u, v, w) ucldlerinin listesi. Bellek: O(m)."""
        return self._edges

    def edge_arrays(self):
        """
        Kenarlari uc ayri numpy dizisi olarak dondurur: (u[], v[], w[]).
        Vektorlestirilmis Bellman-Ford gibi varyantlar icin kullanislidir.
        """
        import numpy as np
        if not self._edges:
            return (np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.float64))
        us = np.fromiter((e[0] for e in self._edges), dtype=np.int64,
                         count=self.m)
        vs = np.fromiter((e[1] for e in self._edges), dtype=np.int64,
                         count=self.m)
        ws = np.fromiter((e[2] for e in self._edges), dtype=np.float64,
                         count=self.m)
        return us, vs, ws

    # ---------------------------------------------------------------- #
    # Gosterim 2: komsuluk listesi                                     #
    # ---------------------------------------------------------------- #
    def adjacency_list(self):
        """
        adj[u] = [(v, w), ...] biciminde komsuluk listesi.
        Bellek: O(n + m). Seyrek graflar icin tercih edilir.
        """
        if self._adj_list is None:
            adj = [[] for _ in range(self.n)]
            for (u, v, w) in self._edges:
                adj[u].append((v, w))
            self._adj_list = adj
        return self._adj_list

    # ---------------------------------------------------------------- #
    # Gosterim 3: komsuluk matrisi                                     #
    # ---------------------------------------------------------------- #
    def adjacency_matrix(self):
        """
        n x n numpy matrisi (float64). Kenar yoksa INF, kosegende 0.
        Cok kenarli durumda en kucuk agirlik tutulur.
        Bellek: O(n^2) -- buyuk seyrek graflarda pahalidir.
        """
        if self._adj_matrix is None:
            import numpy as np
            M = np.full((self.n, self.n), INF, dtype=np.float64)
            for i in range(self.n):
                M[i, i] = 0.0
            for (u, v, w) in self._edges:
                if w < M[u, v]:
                    M[u, v] = w
            self._adj_matrix = M
        return self._adj_matrix

    def matrix_is_feasible(self, max_bytes=2_000_000_000):
        """
        n x n float64 matrisinin bellege sigip sigmadigini tahmin eder.
        Komsuluk matrisi tabanli yaklasimlarin (Dijkstra-dizi, Floyd-Warshall)
        buyuk graflarda neden uygulanamaz oldugunu gostermek icin kullanilir.
        """
        return (self.n * self.n * 8) <= max_bytes

    # ---------------------------------------------------------------- #
    # Yardimci / raporlama                                             #
    # ---------------------------------------------------------------- #
    def density(self):
        """Kenar yogunlugu: m / (n * (n - 1)). 0 (bos) ile 1 (tam) arasi."""
        if self.n <= 1:
            return 0.0
        return self.m / (self.n * (self.n - 1))

    def memory_report(self):
        """
        Uc gosterimin yaklasik bellek ayak izini (bayt) dondurur.
        Komsuluk matrisi henuz uretilmemisse analitik olarak tahmin edilir
        (gereksiz yere O(n^2) bellek ayirmamak icin).
        """
        edge_bytes = deep_getsizeof(self._edges)
        adj_bytes = deep_getsizeof(self.adjacency_list())
        # Matris: 8 bayt/hucre + numpy ust kabugu
        matrix_bytes = self.n * self.n * 8 + 128
        return {
            "edge_list": edge_bytes,
            "adjacency_list": adj_bytes,
            "adjacency_matrix": matrix_bytes,
        }

    def describe(self):
        """Grafin ozet istatistiklerini sozluk olarak dondurur."""
        return {
            "name": self.name,
            "n": self.n,
            "m": self.m,
            "directed": self.directed,
            "density": self.density(),
            "has_negative": self.has_negative,
            "avg_degree": self.m / self.n if self.n else 0.0,
        }

    def __repr__(self):
        return (f"Graph(name={self.name!r}, n={self.n}, m={self.m}, "
                f"directed={self.directed}, density={self.density():.4f})")

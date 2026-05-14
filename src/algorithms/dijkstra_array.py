"""
dijkstra_array.py -- YAKLASIM 2

Dijkstra algoritmasi + lineer dizi taramasi (oncelik kuyrugu YOK) +
komsuluk matrisi.

------------------------------------------------------------------------
TASARIM FELSEFESI : Yapisal olarak basit; yogun graflara optimize.
GOSTERIM          : Komsuluk matrisi -- O(V^2) bellek, ama (u, v) kenar
                    sorgusu O(1).
VERI YAPISI       : Yok. Her adimda en kucuk uzaklikli ziyaret edilmemis
                    dugum, dizinin LINEER TARANMASIYLA bulunur (O(V)).
------------------------------------------------------------------------
ZAMAN KARMASIKLIGI
    O(V^2)
    - V adim yapilir; her adimda min-bulma O(V), gevsetme (matris satiri)
      O(V) -> toplam O(V^2).
    - Bu, kenar sayisindan BAGIMSIZDIR. Yogun graflarda (E ~ V^2)
      O(V^2) < O(E log V) oldugu icin yigin tabanli surumden HIZLIDIR.
    - Seyrek graflarda ise O(V^2), yigin surumunun O(E log V)'sinden
      cok daha yavastir.

BELLEK KARMASIKLIGI
    O(V^2)
    - Komsuluk matrisi. Bu yuzden cok buyuk seyrek graflarda (orn. 264 bin
      dugumlu gercek yol agi) bu yaklasim BELLEGE SIGMAZ -- onemli bir
      pratik kisit.

GRAF YAPISININ ETKISI
    - Yogunluktan neredeyse BAGIMSIZ calisma suresi: hem en buyuk avantaji
      (yogun graf) hem de en buyuk dezavantaji (seyrek graf).
    - NEGATIF kenarlarda Dijkstra varsayimi yine bozulur.

NOT: min-bulma adimi numpy `argmin` ile vektorlestirilmistir. Bu yalnizca
SABIT CARPANI dusurur (Python dongusu yerine C dongusu); algoritmanin
yapisi ve O(V^2) asimptotik karmasikligi DEGISMEZ.
"""
import numpy as np

from config import INF
from algorithms.common import SSSPResult


def dijkstra_array(graph, source, target=None):
    """
    Kaynaktan tum dugumlere en kisa uzakliklari, oncelik kuyrugu
    kullanmadan, komsuluk matrisi uzerinden hesaplar.
    """
    M = graph.adjacency_matrix()          # n x n, kenar yoksa INF, kosegen 0
    n = graph.n

    dist = np.full(n, INF, dtype=np.float64)
    pred = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    dist[source] = 0.0

    pops = 0
    relaxations = 0

    for _ in range(n):
        # 1) Ziyaret edilmemis dugumler arasinda en kucuk uzaklikliyi bul.
        #    visited olanlari INF yaparak elemekten (lineer tarama):
        masked = np.where(visited, INF, dist)
        u = int(np.argmin(masked))
        if masked[u] == INF:
            break                          # kalan dugumler erisilemez
        visited[u] = True
        pops += 1

        if target is not None and u == target:
            break                          # erken durma -- hedefe ulasildi

        # 2) u uzerinden gevsetme. M[u] satiri u'dan giden tum kenarlari
        #    (kenar yoksa INF) tek seferde verir.
        nd = dist[u] + M[u]                 # n uzunlugunda uzaklik vektoru
        improve = (nd < dist) & (~visited)  # nerede gercekten iyilesme var
        relaxations += n                    # her adimda V hucre incelenir
        dist[improve] = nd[improve]
        pred[improve] = u

    return SSSPResult(dist=dist.tolist(), pred=pred.tolist(),
                      source=source, pops=pops, relaxations=relaxations)

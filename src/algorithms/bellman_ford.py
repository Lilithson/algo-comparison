"""
bellman_ford.py -- YAKLASIM 3

Bellman-Ford algoritmasi + kenar listesi.

------------------------------------------------------------------------
TASARIM FELSEFESI : Genellik. Hizdan once dogruluk: negatif agirliklari
                    da dogru cozer, negatif cevrimi tespit eder.
GOSTERIM          : Kenar listesi -- O(E) bellek; algoritma dogal olarak
                    "tum kenarlar uzerinde gez" der, kenar listesi tam
                    olarak bunu verir.
VERI YAPISI       : Yok. Sadece dist/pred dizileri.
------------------------------------------------------------------------
ZAMAN KARMASIKLIGI
    O(V * E)   (en kotu durum)
    - En fazla V-1 tur yapilir; her turda TUM E kenar gevsetilir.
    - "Erken durma" eklenmistir: bir turda hic gevsetme olmazsa algoritma
      durur. Rastgele/seyrek graflarda tur sayisi pratikte V-1'den cok
      kucuktur (kabaca en kisa yollardaki kenar sayisi kadar).

BELLEK KARMASIKLIGI
    O(V + E)
    - Kenar listesi O(E), dist/pred dizileri O(V). Komsuluk matrisine
      gerek yoktur -> buyuk seyrek graflarda Floyd-Warshall'dan ve
      Dijkstra-dizi'den cok daha az bellek kullanir.

GRAF YAPISININ ETKISI
    - O(V * E) carpani yuzunden buyuk graflarda Dijkstra'dan belirgin
      sekilde yavastir; asil degeri HIZ degil GENELLIKTIR.
    - Negatif kenarlar: dogru calisir (Dijkstra'nin calisamadigi yer).
    - Negatif cevrim: tespit edilir ve raporlanir.
"""
from config import INF
from algorithms.common import SSSPResult


def bellman_ford(graph, source, target=None):
    """
    Kaynaktan tum dugumlere en kisa uzakliklari hesaplar. Negatif
    agirliklarla dogru calisir; negatif cevrim varsa bunu isaretler.

    `target` parametresi yalnizca diger algoritmalarla ayni arayuzu
    paylasmak icindir; Bellman-Ford dogasi geregi tek-kaynakli TUM
    uzakliklari hesaplar (erken durma hedefe gore degil, "degisiklik
    olmadi" kosuluna goredir).
    """
    edges = graph.edge_list()
    n = graph.n

    dist = [INF] * n
    pred = [-1] * n
    dist[source] = 0.0

    relaxations = 0
    passes = 0

    # --- ana dongu: en fazla V-1 tur ---
    for _ in range(n - 1):
        passes += 1
        changed = False
        for (u, v, w) in edges:
            du = dist[u]
            if du == INF:
                continue              # u henuz erisilmemis -- gevsetme anlamsiz
            relaxations += 1
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                changed = True
        if not changed:
            break                     # erken durma: bu turda hicbir sey degismedi

    # --- negatif cevrim tespiti ---
    # V-1 turdan sonra hala gevsetilebilen bir kenar varsa, kaynaktan
    # erisilebilen bir negatif cevrim vardir.
    has_neg_cycle = False
    for (u, v, w) in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            has_neg_cycle = True
            break

    return SSSPResult(dist=dist, pred=pred, source=source,
                      has_negative_cycle=has_neg_cycle,
                      relaxations=relaxations, passes=passes)

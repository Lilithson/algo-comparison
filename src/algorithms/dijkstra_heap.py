"""
dijkstra_heap.py -- YAKLASIM 1

Dijkstra algoritmasi + ikili yigin (binary heap) + komsuluk listesi.

------------------------------------------------------------------------
TASARIM FELSEFESI : Talep-uzerine (on-demand), zaman-optimize edilmis.
GOSTERIM          : Komsuluk listesi -- O(V + E) bellek, seyrek graflara uygun.
VERI YAPISI       : Ikili yigin (Python `heapq`) -- en kucuk uzaklikli
                    dugumu O(log V)'de verir.
------------------------------------------------------------------------
ZAMAN KARMASIKLIGI
    O((V + E) log V)
    - Her dugum yigindan en fazla bir kez "kesinlesir" (visited).
    - Her kenar en fazla bir kez gevsetilir; gevsetme yigina bir ekleme
      (push) yapabilir. Toplam push sayisi O(E), her biri O(log V).
    - "Tembel silme" (lazy deletion) kullanildigi icin yigin boyutu
      O(E)'ye kadar cikabilir; bayatlamis girisler cikarildiklarinda
      atlanir.

BELLEK KARMASIKLIGI
    O(V + E)
    - Komsuluk listesi O(V + E), dist/pred dizileri O(V), yigin O(E).

GRAF YAPISININ ETKISI
    - Seyrek graflarda (E = O(V)) calisma suresi ~ O(V log V): cok hizli.
    - Yogun graflarda (E = O(V^2)) ~ O(V^2 log V): bu durumda O(V^2)
      olan dizi-tabanli surum daha iyidir.
    - NEGATIF kenarlarda dogru calismaz: bir dugum "kesinlestikten" sonra
      daha kisa bir yol bulunabilir; Dijkstra'nin acgozlu varsayimi bozulur.
"""
import heapq

from config import INF
from algorithms.common import SSSPResult


def dijkstra_heap(graph, source, target=None):
    """
    Kaynaktan tum dugumlere en kisa uzakliklari hesaplar.

    target verilirse, hedef dugum yigindan cikarildigi anda durulur.
    Bu "erken durma" tek-cift (s, t) sorgularinda ortalamada isin bir
    kismini atlar -- talep-uzerine optimizasyonun en yalin ornegi.
    """
    adj = graph.adjacency_list()
    n = graph.n

    dist = [INF] * n
    pred = [-1] * n
    visited = [False] * n
    dist[source] = 0.0

    # Yigin elemanlari: (uzaklik, dugum). Python heapq bir min-yigindir.
    heap = [(0.0, source)]

    pops = 0          # yigindan kesinlesen dugum sayisi
    relaxations = 0   # gevsetme denemesi sayisi

    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            # Bayatlamis giris: u zaten daha kisa bir uzaklikla kesinlesti.
            continue
        visited[u] = True
        pops += 1

        if target is not None and u == target:
            break     # erken durma -- hedefe ulasildi

        # u'nun komsularini gevset
        for (v, w) in adj[u]:
            relaxations += 1
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))

    return SSSPResult(dist=dist, pred=pred, source=source,
                      pops=pops, relaxations=relaxations)

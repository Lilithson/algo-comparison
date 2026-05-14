"""
common.py -- dort algoritmanin ortak kullandigi sonuc tipleri ve
yardimci fonksiyonlar.

Tum algoritmalarin ayni arayuzu paylasmasi, benchmark ve dogrulama
kodunun onlara tek bicimde davranabilmesini saglar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config import INF


@dataclass
class SSSPResult:
    """
    Tek-kaynakli en-kisa-yol (Single Source Shortest Path) sonucu.
    Dijkstra (her iki surum) ve Bellman-Ford bunu dondurur.

    dist[v]  : kaynaktan v'ye en kisa uzaklik (erisilemezse INF).
    pred[v]  : en kisa yol agacinda v'nin atasi (-1 ise yok).
    """
    dist: List[float]
    pred: List[int]
    source: int
    has_negative_cycle: bool = False
    # --- analiz icin sayaclar (zamanlamayi etkilemeyecek kadar ucuz) ---
    relaxations: int = 0     # kac kez kenar gevsetme denemesi yapildi
    pops: int = 0            # yigindan / dizi taramasindan kac dugum kesinlesti
    passes: int = 0          # Bellman-Ford: kac tam tur yapildi

    def distance_to(self, target):
        return self.dist[target]


@dataclass
class APSPResult:
    """
    Tum-ciftler en-kisa-yol (All Pairs Shortest Path) sonucu.
    Floyd-Warshall bunu dondurur.

    dist : n x n numpy matrisi; dist[i, j] = i'den j'ye en kisa uzaklik.
    nxt  : n x n "sonraki dugum" matrisi; yol geri kurma icin.
    """
    dist: object             # numpy.ndarray (n x n) veya liste-listesi
    nxt: object              # numpy.ndarray (n x n) veya liste-listesi
    has_negative_cycle: bool = False
    preprocess_relaxations: int = 0

    def distance_to(self, source, target):
        # dist[s][t] indekslemesi hem numpy matrisinde hem de
        # liste-listesinde calisir (vektorlestirilmis ve referans surum).
        return float(self.dist[source][target])


def query_distance(result, source, target):
    """
    Bir algoritma sonucundan (s, t) sorgusunun cevabini tek bicimde okur.
    SSSPResult ise dist[t]; APSPResult ise dist[s][t].
    """
    if isinstance(result, APSPResult):
        return float(result.dist[source][target])
    return result.dist[target]


def reconstruct_path(pred, source, target):
    """
    Tek-kaynakli algoritmalarin `pred` dizisinden source -> target
    yolunu kurar. Yol yoksa bos liste doner.
    """
    if source == target:
        return [source]
    path = []
    v = target
    guard = 0
    limit = len(pred) + 1
    while v != -1 and guard <= limit:
        path.append(v)
        if v == source:
            path.reverse()
            return path
        v = pred[v]
        guard += 1
    return []   # erisilemez (veya beklenmeyen durum)


def reconstruct_path_fw(nxt, source, target):
    """
    Floyd-Warshall'in `nxt` matrisinden source -> target yolunu kurar.
    Yol yoksa bos liste doner.
    """
    if source == target:
        return [source]
    if int(nxt[source][target]) < 0:
        return []
    path = [source]
    u = source
    guard = 0
    limit = len(nxt) + 1
    while u != target and guard <= limit:
        u = int(nxt[u][target])
        if u < 0:
            return []
        path.append(u)
        guard += 1
    return path if u == target else []


def path_cost(graph, path):
    """Verilen bir yolun toplam agirligini hesaplar (dogrulama icin)."""
    if len(path) <= 1:
        return 0.0 if path else INF
    # kenar agirliklarini hizli aramak icin sozluk kur
    weight = {}
    for (u, v, w) in graph.edge_list():
        key = (u, v)
        if key not in weight or w < weight[key]:
            weight[key] = w
    total = 0.0
    for a, b in zip(path, path[1:]):
        if (a, b) not in weight:
            return INF   # gecersiz yol
        total += weight[(a, b)]
    return total

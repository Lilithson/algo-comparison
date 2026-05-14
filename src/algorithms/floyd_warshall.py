"""
floyd_warshall.py -- YAKLASIM 4

Floyd-Warshall algoritmasi (TUM ciftler arasi en kisa yollar) +
komsuluk matrisi.

------------------------------------------------------------------------
TASARIM FELSEFESI : Agir on-isleme, cok hizli sorgu. Bir kez O(V^3)
                    harcanir; sonrasinda her (s, t) sorgusu O(1).
GOSTERIM          : Komsuluk matrisi -- algoritma dogrudan matris
                    uzerinde calisir.
VERI YAPISI       : Iki adet n x n matris (uzakliklar + yol geri kurma).
------------------------------------------------------------------------
ZAMAN KARMASIKLIGI
    On-isleme : O(V^3)   -- her ara dugum k icin tum (i, j) ciftleri
                            guncellenir.
    Sorgu     : O(1)     -- yalnizca matris erisimi.
    Q sorgu icin toplam: O(V^3 + Q).  Dijkstra ile karsilastir: O(Q*(E log V)).
    Yeterince cok sorgu yapilacaksa (buyuk Q) Floyd-Warshall karli hale gelir.

BELLEK KARMASIKLIGI
    O(V^2)
    - Uzaklik matrisi + "sonraki dugum" matrisi. Buyuk seyrek graflarda
      (orn. 264 bin dugum -> ~70 milyar hucre) BELLEGE SIGMAZ.

GRAF YAPISININ ETKISI
    - Calisma suresi yalnizca V'ye baglidir; yogunluktan ve kenar
      sayisindan tamamen BAGIMSIZDIR.
    - Negatif kenarlarla calisir; negatif cevrimi kosegende negatif
      deger olarak tespit eder.
    - Tek bir sorgu icin korkunc derecede israftir; cok sayida sorgu
      icin idealdir.

NOT: en ictteki iki dongu numpy ile vektorlestirilmistir (yayinim /
broadcasting). Bu yalnizca SABIT CARPANI dusurur; uc katmanli yapidan
gelen O(V^3) asimptotik karmasiklik DEGISMEZ. Algoritmanin "ders kitabi"
hali asagidaki `floyd_warshall_reference` fonksiyonunda saf Python ile
verilmistir.
"""
import numpy as np

from config import INF
from algorithms.common import APSPResult


def floyd_warshall(graph, with_paths=True):
    """
    Tum dugum ciftleri arasindaki en kisa uzakliklari hesaplar
    (vektorlestirilmis surum -- benchmark bu surumu kullanir).
    """
    M = graph.adjacency_matrix()
    n = graph.n
    D = M.copy()                           # uzaklik matrisi (calisma kopyasi)

    if with_paths:
        # nxt[i, j] = i'den j'ye en kisa yolda i'den hemen SONRAKI dugum.
        # Baslangic: dogrudan kenar varsa nxt[i, j] = j, yoksa -1.
        idx_j = np.tile(np.arange(n, dtype=np.int32), (n, 1))   # idx_j[i, j] = j
        direct = np.isfinite(M)            # kosegen + gercek kenarlar
        np.fill_diagonal(direct, False)    # kosegeni cikar -> sadece kenarlar
        nxt = np.where(direct, idx_j, np.int32(-1)).astype(np.int32)
    else:
        nxt = None

    # --- ana dongu: her ara dugum k icin ---
    for k in range(n):
        # i -> k -> j yolunun maliyeti (n x n matris, yayinim ile)
        through_k = D[:, k][:, None] + D[k, :][None, :]
        improved = through_k < D
        # D = min(D, through_k)  -- yerinde (in-place), gereksiz tahsis yok
        np.minimum(D, through_k, out=D)
        if with_paths:
            # Iyilesen (i, j) icin yol once k yonune gider:
            # nxt[i, j] = nxt[i, k]. (k. sutun bu turda degismez.)
            col_k = nxt[:, k].copy()
            np.copyto(nxt, col_k[:, None], where=improved)

    # Negatif cevrim: kosegende negatif bir deger olusmussa cevrim vardir.
    has_neg_cycle = bool(np.any(np.diag(D) < 0))

    return APSPResult(dist=D, nxt=nxt, has_negative_cycle=has_neg_cycle,
                      preprocess_relaxations=n * n * n)


def floyd_warshall_reference(graph):
    """
    Floyd-Warshall'in DERS KITABI surumu -- saf Python, uc katmanli dongu.

    Vektorlestirilmis surumle ayni sonucu uretir; algoritmanin yapisini
    en net haliyle gosterir. O(V^3) oldugu icin yalnizca kucuk graflarda
    pratiktir ve esas olarak dogrulama (verify) icin kullanilir.
    """
    M = graph.adjacency_matrix()
    n = graph.n
    # Matrisi saf Python liste-listesine cevir
    D = [[float(M[i][j]) for j in range(n)] for i in range(n)]
    nxt = [[-1] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and D[i][j] < INF:
                nxt[i][j] = j

    for k in range(n):
        Dk = D[k]
        for i in range(n):
            Di = D[i]
            dik = Di[k]
            if dik == INF:
                continue               # i'den k'ye yol yok -> k uzerinden gecemez
            nxt_ik = nxt[i][k]
            for j in range(n):
                nd = dik + Dk[j]
                if nd < Di[j]:
                    Di[j] = nd
                    nxt[i][j] = nxt_ik

    has_neg_cycle = any(D[i][i] < 0 for i in range(n))
    return APSPResult(dist=D, nxt=nxt, has_negative_cycle=has_neg_cycle,
                      preprocess_relaxations=n * n * n)

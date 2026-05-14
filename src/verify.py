"""
verify.py -- DOGRULAMA (correctness verification) paketi.

"Correctness alone is not sufficient for full credit" -- ama once
dogruluk gelir. Bu betik dort yaklasimin DOGRU calistigini bagimsiz
yontemlerle kanitlar:

  [1] El ile hesaplanmis kucuk bir ornek (beklenen cevap bilinir).
  [2] Dort algoritmanin + Floyd-Warshall referans (saf Python) surumunun
      cok sayida rastgele graf turunde BIREBIR ayni uzakliklari uretmesi.
  [3] Yol geri kurmanin dogrulugu: geri kurulan yolun maliyeti, raporlanan
      uzakliga esit olmali.
  [4] Negatif kenarlarda dogruluk: Bellman-Ford ve Floyd-Warshall dogru
      sonucu verir; Dijkstra'nin neden veremedigi gosterilir.
  [5] Negatif cevrim tespiti: Bellman-Ford ve Floyd-Warshall negatif
      cevrimi yakalar.

Calistirma:  python src/verify.py
Cikis kodu:  tum testler gecerse 0, aksi halde 1.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import INF
from graph import Graph
from generators import random_sparse, random_dense, grid_2d, scale_free, random_dag
from algorithms import (
    dijkstra_heap, dijkstra_array, bellman_ford, floyd_warshall,
    query_distance, reconstruct_path, reconstruct_path_fw,
)
from algorithms.floyd_warshall import floyd_warshall_reference
from algorithms.common import path_cost


def _eq(a, b, tol=1e-6):
    """INF degerlerini de dogru karsilastiran yaklasik esitlik."""
    if a == b:                       # inf == inf veya tam esitlik
        return True
    if a == INF or b == INF:
        return False
    return abs(a - b) <= tol


# --------------------------------------------------------------------------
# [1] El ile hesaplanmis kucuk ornek
# --------------------------------------------------------------------------
def test_hand_example():
    """
    5 dugumlu, elle cozulebilen yonlu graf. Kaynak = 0.
      0->1:10  0->2:3  2->1:4  1->3:2  2->3:8  2->4:2  3->4:7  4->3:9
    Beklenen en kisa uzakliklar (kaynak 0):
      0:0, 1:7 (0-2-1), 2:3 (0-2), 3:9 (0-2-1-3), 4:5 (0-2-4)
    """
    edges = [(0, 1, 10), (0, 2, 3), (2, 1, 4), (1, 3, 2),
             (2, 3, 8), (2, 4, 2), (3, 4, 7), (4, 3, 9)]
    g = Graph(5, edges, directed=True, name="el-ornegi")
    expected = [0, 7, 3, 9, 5]

    results = {
        "dijkstra_heap": dijkstra_heap(g, 0),
        "dijkstra_array": dijkstra_array(g, 0),
        "bellman_ford": bellman_ford(g, 0),
        "floyd_warshall": floyd_warshall(g),
        "floyd_warshall_reference": floyd_warshall_reference(g),
    }
    ok = True
    lines = []
    for name, res in results.items():
        got = [query_distance(res, 0, t) for t in range(5)]
        match = all(_eq(got[t], expected[t]) for t in range(5))
        ok = ok and match
        lines.append(f"      {name:28s} -> {got}  {'OK' if match else 'HATA'}")
    detail = (f"    Beklenen uzakliklar: {expected}\n" + "\n".join(lines))
    return ok, detail


# --------------------------------------------------------------------------
# [2] Dort algoritmanin capraz kontrolu
# --------------------------------------------------------------------------
def test_cross_check():
    """
    Cesitli graf turlerinde (seyrek, yogun, izgara, olceksiz) dort
    algoritma + Floyd-Warshall referans surumu BIREBIR ayni uzakliklari
    uretmeli. Graflar Floyd-Warshall referansinin (saf Python O(V^3))
    hizli kalmasi icin kucuk tutulur.
    """
    cases = []
    for seed in range(4):
        cases.append(random_sparse(20 + 6 * seed, avg_degree=4, seed=seed))
        cases.append(random_dense(15 + 5 * seed, edge_frac=0.35, seed=seed))
        cases.append(grid_2d(4 + seed, 5 + seed, seed=seed))
        cases.append(scale_free(20 + 6 * seed, m_attach=2, seed=seed))

    total_pairs = 0
    total_graphs = 0
    mismatches = []

    for g in cases:
        total_graphs += 1
        # Floyd-Warshall: tek kosuda tum kaynaklar
        fw = floyd_warshall(g)
        fw_ref = floyd_warshall_reference(g)
        # birkac kaynaktan tek-kaynakli algoritmalar
        sources = sorted(set([0, g.n // 3, g.n // 2, g.n - 1]))
        for s in sources:
            rh = dijkstra_heap(g, s)
            ra = dijkstra_array(g, s)
            rb = bellman_ford(g, s)
            for t in range(g.n):
                total_pairs += 1
                ref = rh.dist[t]
                vals = {
                    "dijkstra_array": ra.dist[t],
                    "bellman_ford": rb.dist[t],
                    "floyd_warshall": query_distance(fw, s, t),
                    "floyd_warshall_ref": query_distance(fw_ref, s, t),
                }
                for name, v in vals.items():
                    if not _eq(ref, v):
                        mismatches.append(
                            f"{g.name} s={s} t={t}: dijkstra_heap={ref} "
                            f"!= {name}={v}")

    ok = not mismatches
    detail = (f"    {total_graphs} graf, {total_pairs} (kaynak,hedef) cifti "
              f"karsilastirildi.")
    if mismatches:
        detail += "\n    UYUSMAZLIKLAR:\n      " + "\n      ".join(mismatches[:8])
    else:
        detail += "\n    Tum algoritmalar her ciftte birebir ayni sonucu verdi."
    return ok, detail


# --------------------------------------------------------------------------
# [3] Yol geri kurma dogrulamasi
# --------------------------------------------------------------------------
def test_path_reconstruction():
    """
    Geri kurulan yolun gercekten kaynaktan hedefe gittigini ve toplam
    agirliginin raporlanan uzakliga esit oldugunu dogrular.
    """
    checks = 0
    bad = []
    for seed in range(5):
        g = random_sparse(60, avg_degree=4, seed=seed)
        rh = dijkstra_heap(g, 0)
        fw = floyd_warshall(g)
        for t in range(g.n):
            d = rh.dist[t]
            if d == INF:
                continue
            checks += 1
            # Dijkstra (pred) yolu
            p = reconstruct_path(rh.pred, 0, t)
            if not p or p[0] != 0 or p[-1] != t or not _eq(path_cost(g, p), d):
                bad.append(f"{g.name}: dijkstra yol 0->{t} hatali: {p}")
            # Floyd-Warshall (nxt) yolu
            pf = reconstruct_path_fw(fw.nxt, 0, t)
            if not pf or pf[0] != 0 or pf[-1] != t or not _eq(path_cost(g, pf), d):
                bad.append(f"{g.name}: FW yol 0->{t} hatali: {pf}")
    ok = not bad
    detail = f"    {checks} yol geri kuruldu ve maliyeti uzaklikla doğrulandi."
    if bad:
        detail += "\n    HATALAR:\n      " + "\n      ".join(bad[:6])
    return ok, detail


# --------------------------------------------------------------------------
# [4] Negatif kenarlarda dogruluk
# --------------------------------------------------------------------------
def test_negative_weights():
    """
    Negatif kenar iceren (ama negatif cevrimi olmayan) graf.

    Kucuk, elle dogrulanabilir ornek:
      0->2:2  2->3:5  0->1:3  1->2:-2
    Gercek en kisa uzakliklar (kaynak 0): 0:0, 1:3, 2:1 (0-1-2), 3:6 (0-1-2-3)

    Bellman-Ford ve Floyd-Warshall DOGRU sonucu verir.
    Dijkstra (yigin) ise dugum 2'yi erken kesinlestirdigi icin dist[3]'u
    yanlis (7) hesaplar -- bu, Dijkstra'nin negatif kenar kisitinin somut
    kanitidir.
    """
    edges = [(0, 2, 2), (2, 3, 5), (0, 1, 3), (1, 2, -2)]
    g = Graph(4, edges, directed=True, name="negatif-ornek")
    correct = [0, 3, 1, 6]

    rb = bellman_ford(g, 0)
    fw = floyd_warshall(g)
    fw_ref = floyd_warshall_reference(g)
    rh = dijkstra_heap(g, 0)

    bf_ok = all(_eq(rb.dist[t], correct[t]) for t in range(4))
    fw_ok = all(_eq(query_distance(fw, 0, t), correct[t]) for t in range(4))
    fwr_ok = all(_eq(query_distance(fw_ref, 0, t), correct[t]) for t in range(4))
    dij = [rh.dist[t] for t in range(4)]
    dij_wrong = not all(_eq(dij[t], correct[t]) for t in range(4))

    # DAG uzerinde de Bellman-Ford ile Floyd-Warshall uyumlu olmali
    gd = random_dag(40, avg_degree=4, allow_negative=True, seed=7)
    rbf = bellman_ford(gd, 0)
    rfw = floyd_warshall(gd)
    dag_ok = all(_eq(rbf.dist[t], query_distance(rfw, 0, t))
                 for t in range(gd.n))

    ok = bf_ok and fw_ok and fwr_ok and dag_ok
    detail = (
        f"    Gercek uzakliklar (kaynak 0): {correct}\n"
        f"      Bellman-Ford   -> {[rb.dist[t] for t in range(4)]}  "
        f"{'OK' if bf_ok else 'HATA'}\n"
        f"      Floyd-Warshall -> {[query_distance(fw,0,t) for t in range(4)]}  "
        f"{'OK' if fw_ok else 'HATA'}\n"
        f"      Dijkstra-yigin -> {dij}  "
        f"{'(beklendigi gibi YANLIS -- negatif kenar kisiti)' if dij_wrong else 'BEKLENMEDIK'}\n"
        f"    Negatif agirlikli DAG'da Bellman-Ford = Floyd-Warshall: "
        f"{'OK' if dag_ok else 'HATA'}")
    # Dijkstra'nin yanlis olmasi da dogrulamanin bir parcasi: kisiti gosteriyoruz
    return ok and dij_wrong, detail


# --------------------------------------------------------------------------
# [5] Negatif cevrim tespiti
# --------------------------------------------------------------------------
def test_negative_cycle():
    """
    Kaynaktan erisilebilen bir negatif cevrim:
      0->1:1, 1->2:-1, 2->3:-1, 3->1:-1   (cevrim 1->2->3->1 toplami -3)
    Bellman-Ford ve Floyd-Warshall bunu tespit etmeli.
    Ek olarak: negatif cevrimi OLMAYAN bir graf yanlislikla
    isaretlenmemeli.
    """
    edges = [(0, 1, 1), (1, 2, -1), (2, 3, -1), (3, 1, -1)]
    g = Graph(4, edges, directed=True, name="negatif-cevrim")
    rb = bellman_ford(g, 0)
    fw = floyd_warshall(g)
    detected = rb.has_negative_cycle and fw.has_negative_cycle

    # negatif cevrimi olmayan kontrol grafi
    g2 = random_sparse(50, avg_degree=4, seed=11)
    rb2 = bellman_ford(g2, 0)
    fw2 = floyd_warshall(g2)
    no_false_alarm = (not rb2.has_negative_cycle) and (not fw2.has_negative_cycle)

    ok = detected and no_false_alarm
    detail = (
        f"    Negatif cevrimli graf: Bellman-Ford tespit={rb.has_negative_cycle}, "
        f"Floyd-Warshall tespit={fw.has_negative_cycle}  "
        f"{'OK' if detected else 'HATA'}\n"
        f"    Temiz graf (yanlis alarm olmamali): Bellman-Ford={rb2.has_negative_cycle}, "
        f"Floyd-Warshall={fw2.has_negative_cycle}  "
        f"{'OK' if no_false_alarm else 'HATA'}")
    return ok, detail


# --------------------------------------------------------------------------
# Ana akis
# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  DOGRULAMA  --  En-Kisa-Yol Algoritmalari Dogruluk Testleri")
    print("=" * 70)

    tests = [
        ("[1] El ile hesaplanmis kucuk ornek", test_hand_example),
        ("[2] Dort algoritmanin capraz kontrolu (rastgele graflar)",
         test_cross_check),
        ("[3] Yol geri kurma dogrulamasi", test_path_reconstruction),
        ("[4] Negatif kenarlarda dogruluk (BF/FW dogru, Dijkstra degil)",
         test_negative_weights),
        ("[5] Negatif cevrim tespiti", test_negative_cycle),
    ]

    passed = 0
    for title, fn in tests:
        print(f"\n{title}")
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"    ISTISNA: {exc!r}"
        print(detail)
        print(f"    --> {'GECTI' if ok else 'KALDI'}")
        passed += int(ok)

    print("\n" + "=" * 70)
    status = "TUM TESTLER GECTI" if passed == len(tests) else "BAZI TESTLER KALDI"
    print(f"  SONUC: {status}  ({passed}/{len(tests)})")
    print("=" * 70)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())

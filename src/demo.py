# -*- coding: utf-8 -*-
"""
demo.py -- CANLI KARSILASTIRMA DEMOSU

Dort en-kisa-yol yaklasimini, secilen bir graf uzerinde AYNI (kaynak, hedef)
sorgusu icin yan yana calistirir; bulunan uzakligi, calisma suresini, yapilan
is miktarini (gevsetme sayisi) ve tepe bellegi tek bir tabloda gosterir.

Sunum sirasinda canli calistirilabilir; tek bir komutla dort yaklasimin
"ayni cevabi farkli maliyetle" urettigi gorulur.

Kullanim ornekleri:
  python src/demo.py                                  # varsayilan: 20x20 izgara
  python src/demo.py --kind grid --n 900               # daha buyuk izgara
  python src/demo.py --kind sparse --n 2000             # seyrek rastgele graf
  python src/demo.py --kind real  --n 800               # gercek yol agindan alt-graf
  python src/demo.py --kind grid --n 25 --trace         # kucuk graf + adim adim iz
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import INF, GLOBAL_SEED
import generators as gen
from data_loader import load_dimacs, sample_connected_subgraph, ensure_dimacs_ny
from algorithms import (dijkstra_heap, dijkstra_array, bellman_ford,
                        floyd_warshall, query_distance, reconstruct_path,
                        reconstruct_path_fw)
from algorithms.common import path_cost

# Komsuluk matrisi tabanli yaklasimlarin demo icin ust siniri (bellek + hiz)
MATRIX_NODE_LIMIT = 1500


def build_graph(kind, n, seed):
    """Istenen turde bir graf uretir veya gercek veriden alt-graf orneklar."""
    if kind == "grid":
        return gen.grid_for_size(n, seed=seed)
    if kind == "sparse":
        return gen.random_sparse(n, seed=seed)
    if kind == "dense":
        return gen.random_dense(n, edge_frac=0.3, seed=seed)
    if kind == "real":
        ensure_dimacs_ny()
        print("  (gercek yol agi yukleniyor...)")
        full = load_dimacs()
        return sample_connected_subgraph(full, n, seed=seed)
    raise ValueError(f"bilinmeyen graf turu: {kind!r}")


def measure(fn):
    """fn'i calistirir; (sonuc, sure_sn, tepe_bellek_bayt) dondurur."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def fmt_time(s):
    if s >= 1.0:
        return f"{s:8.2f} s "
    if s >= 1e-3:
        return f"{s*1e3:8.2f} ms"
    return f"{s*1e6:8.1f} us"


def fmt_mem(b):
    if b >= 1024 * 1024:
        return f"{b/1024/1024:7.1f} MB"
    return f"{b/1024:7.1f} KB"


def run_all(graph, src, dst):
    """Dort algoritmayi calistirir; her biri icin olcum sozlugu dondurur."""
    n = graph.n
    rows = []

    # 1) Dijkstra + ikili yigin
    res, t, mem = measure(lambda: dijkstra_heap(graph, src))
    rows.append({"ad": "Dijkstra-Yigin", "uzaklik": res.dist[dst], "sure": t,
                 "is": res.relaxations, "is_adi": "gevsetme",
                 "bellek": mem, "durum": "calisti", "pred": res.pred,
                 "kind": "sssp"})

    # 2) Dijkstra + dizi tarama (komsuluk matrisi gerektirir)
    if n <= MATRIX_NODE_LIMIT:
        res, t, mem = measure(lambda: dijkstra_array(graph, src))
        rows.append({"ad": "Dijkstra-Dizi", "uzaklik": res.dist[dst], "sure": t,
                     "is": res.relaxations, "is_adi": "inceleme",
                     "bellek": mem, "durum": "calisti", "pred": res.pred,
                     "kind": "sssp"})
    else:
        rows.append({"ad": "Dijkstra-Dizi", "durum": "atlandi",
                     "neden": f"komsuluk matrisi cok buyuk (V>{MATRIX_NODE_LIMIT})",
                     "kind": "sssp"})

    # 3) Bellman-Ford
    res, t, mem = measure(lambda: bellman_ford(graph, src))
    rows.append({"ad": "Bellman-Ford", "uzaklik": res.dist[dst], "sure": t,
                 "is": res.relaxations, "is_adi": "gevsetme",
                 "bellek": mem, "durum": "calisti", "pred": res.pred,
                 "passes": res.passes, "kind": "sssp"})

    # 4) Floyd-Warshall (komsuluk matrisi + O(V^3))
    if n <= MATRIX_NODE_LIMIT:
        res, t, mem = measure(lambda: floyd_warshall(graph))
        rows.append({"ad": "Floyd-Warshall",
                     "uzaklik": query_distance(res, src, dst), "sure": t,
                     "is": n * n * n, "is_adi": "guncelleme",
                     "bellek": mem, "durum": "calisti", "nxt": res.nxt,
                     "kind": "apsp"})
    else:
        rows.append({"ad": "Floyd-Warshall", "durum": "atlandi",
                     "neden": f"O(V^2) bellek + O(V^3) zaman (V>{MATRIX_NODE_LIMIT})",
                     "kind": "apsp"})
    return rows


def print_table(rows):
    print()
    header = (f"{'Algoritma':<16}{'Uzaklik':>11}{'Sure':>13}  "
              f"{'Yapilan is':<24}{'Tepe bellek':>13}")
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["durum"] == "atlandi":
            print(f"{r['ad']:<16}{'UYGULANAMAZ':>11}   {r['neden']}")
            continue
        work = f"{r['is']:,} {r['is_adi']}"
        print(f"{r['ad']:<16}{r['uzaklik']:>11.1f}{fmt_time(r['sure']):>13}  "
              f"{work:<24}{fmt_mem(r['bellek']):>13}")
    print("-" * len(header))


def print_observations(rows, graph, src, dst):
    ran = [r for r in rows if r["durum"] == "calisti"]
    # dogruluk kontrolu
    dists = {round(r["uzaklik"], 6) for r in ran}
    if len(dists) == 1:
        d = dists.pop()
        if d == INF:
            print(f"\n[!] Hedef {dst}, kaynak {src}'tan ERISILEMEZ "
                  f"(tum algoritmalar hemfikir).")
            return
        print(f"\n[OK] Dort algoritma da AYNI en kisa uzakligi buldu: {d:g}")
    else:
        print(f"\n[HATA] Algoritmalar farkli sonuc verdi: {dists}")

    # yol (Dijkstra-Yigin'in pred'inden)
    dh = next(r for r in ran if r["ad"] == "Dijkstra-Yigin")
    path = reconstruct_path(dh["pred"], src, dst)
    if path:
        if len(path) <= 12:
            yol = " -> ".join(map(str, path))
        else:
            yol = (" -> ".join(map(str, path[:5])) + " -> ... -> "
                   + " -> ".join(map(str, path[-4:])))
        print(f"     Bulunan yol ({len(path)} dugum, {len(path)-1} kenar): {yol}")

    # gozlemler
    print("\nGozlemler:")
    fastest = min(ran, key=lambda r: r["sure"])
    slowest = max(ran, key=lambda r: r["sure"])
    print(f"  - En hizli : {fastest['ad']}  ({fmt_time(fastest['sure']).strip()})")
    print(f"  - En yavas : {slowest['ad']}  ({fmt_time(slowest['sure']).strip()})")
    if fastest is not slowest and fastest["sure"] > 0:
        print(f"  - {slowest['ad']}, {fastest['ad']}'dan "
              f"{slowest['sure']/fastest['sure']:,.0f}x daha yavas.")
    # is miktari karsilastirmasi
    dh_is = dh["is"]
    da = next((r for r in ran if r["ad"] == "Dijkstra-Dizi"), None)
    if da:
        print(f"  - Dijkstra-Dizi {da['is']:,} hucre inceledi; "
              f"Dijkstra-Yigin yalnizca {dh_is:,} kenar gevsetti "
              f"(ayni algoritma, farkli veri yapisi).")
    fw = next((r for r in ran if r["ad"] == "Floyd-Warshall"), None)
    if fw:
        print(f"  - Floyd-Warshall {fw['is']:,} guncelleme yapti: tek bir "
              f"sorgu icin TUM ciftleri hesapladi (asiri is).")
    bf = next((r for r in ran if r["ad"] == "Bellman-Ford"), None)
    if bf and "passes" in bf:
        print(f"  - Bellman-Ford {bf['passes']} turda yakinsadi "
              f"(erken durma; en kotu durum V-1={graph.n-1} tur).")


def print_trace(graph, src, dst):
    """
    Adim adim iz -- her graf boyutunda calisir:
      * Dijkstra : dugumleri hangi sirayla kesinlestirdigi (buyuk graflarda
                   ozetlenir) + hedefe kacinci adimda ulastigi (erken durma).
      * Bellman-Ford : her turda kac kenar gevsetildigi, kac tur surdugu.
    """
    import heapq
    n = graph.n
    adj = graph.adjacency_list()
    edges = graph.edge_list()
    print("\n" + "=" * 60)
    print("  ADIM ADIM IZ")
    print("=" * 60)

    # --- Dijkstra: kesinlestirme sirasi ---
    dist = [INF] * n
    dist[src] = 0.0
    heap = [(0.0, src)]
    visited = [False] * n
    order = []
    target_rank = None
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        if u == dst and target_rank is None:
            target_rank = len(order)
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))

    print("\nDijkstra-Yigin -- dugumleri kesinlestirme sirasi (uzakliga gore):")
    if len(order) <= 30:
        print("  " + " -> ".join(map(str, order)))
    else:
        head = " -> ".join(map(str, order[:14]))
        print(f"  {head} -> ... -> {order[-1]}   "
              f"(toplam {len(order)} dugum kesinlesti)")
    print("  (her adimda o ana kadarki EN YAKIN kesinlesmemis dugum secilir)")
    if target_rank is not None:
        print(f"  Hedef dugum {dst}, {target_rank}. sirada kesinlesti "
              f"(uzaklik = {dist[dst]:g}).")
        if target_rank < n:
            print(f"  -> ERKEN DURMA ile yalnizca {target_rank}/{n} dugum "
                  f"acmak yeterdi; kalan {n - target_rank} dugum gereksizdi.")
    else:
        print(f"  Hedef dugum {dst}, kaynak {src}'tan erisilemez.")

    # --- Bellman-Ford: tur tur ---
    dist = [INF] * n
    dist[src] = 0.0
    print("\nBellman-Ford -- tur tur yakinsama:")
    shown = 0
    for p in range(1, n):
        changed = 0
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed += 1
        finite = sum(1 for x in dist if x != INF)
        if p <= 25:
            print(f"  Tur {p:2d}: TUM {len(edges)} kenar tarandi, "
                  f"{changed:5d} gevsetme; erisilen dugum = {finite}/{n}")
            shown = p
        if changed == 0:
            if p > shown:
                print(f"  ... (Tur {shown+1}-{p} arasi gosterilmedi)")
            print(f"  -> Tur {p}'de hic gevsetme yok: ERKEN DURMA "
                  f"({p} tur / en kotu durum {n-1} tur).")
            break

    print("\nFark: Dijkstra hedefe dogru SECEREK ilerler ve hedefe ulasinca "
          "durabilir;\n      Bellman-Ford ise her turda AYRIM yapmadan TUM "
          "kenarlari yeniden tarar.")


def main():
    parser = argparse.ArgumentParser(
        description="En kisa yol -- dort yaklasimin canli karsilastirmasi")
    parser.add_argument("--kind", default="grid",
                        choices=["grid", "sparse", "dense", "real"],
                        help="graf turu (varsayilan: grid)")
    parser.add_argument("--n", type=int, default=400,
                        help="dugum sayisi (varsayilan: 400)")
    parser.add_argument("--src", type=int, default=0, help="kaynak dugum")
    parser.add_argument("--dst", type=int, default=-1,
                        help="hedef dugum (-1 = son dugum)")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="tohum")
    parser.add_argument("--trace", action="store_true",
                        help="kucuk graflar icin adim adim iz goster")
    args = parser.parse_args()

    print("=" * 60)
    print("  EN KISA YOL  --  CANLI KARSILASTIRMA DEMOSU")
    print("=" * 60)

    graph = build_graph(args.kind, args.n, args.seed)
    src = args.src % graph.n
    dst = args.dst % graph.n
    print(f"Graf   : {graph.name}")
    print(f"         V = {graph.n:,}   E = {graph.m:,}   "
          f"yogunluk = {graph.density():.5f}")
    print(f"Sorgu  : kaynak = {src}   hedef = {dst}")

    rows = run_all(graph, src, dst)
    print_table(rows)
    print_observations(rows, graph, src, dst)

    if args.trace:
        print_trace(graph, src, dst)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

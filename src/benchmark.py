"""
benchmark.py -- DENEYSEL DEGERLENDIRME kosucusu.

Dort en-kisa-yol yaklasimini KARSILASTIRMALI olarak olcer. Her olcumde
hem calisma suresi (time.perf_counter, birden cok kez, medyan) hem de
TEPE BELLEK (tracemalloc) kaydedilir.

Deneyler:
  [A] Olceklenme  : seyrek graflarda dugum sayisi (V) buyudukce sure.
  [B] Yogunluk    : V sabit, kenar yogunlugu degisirken sure.
  [C] Sorgu sayisi: on-isleme / sorgu odunlesimi (Q sorgu icin toplam sure).
  [D] Gercek ag   : DIMACS New York yol agi -- hem ornek alt-graflar
                    (dort algoritma) hem de tam graf (264 bin dugum).

Calistirma:
  python src/benchmark.py --quick   (hizli, ~yarim dakika)
  python src/benchmark.py --full    (rapor sonuclari, birkac dakika)

Sonuc: results/raw/benchmark_results.json
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import random
import statistics
import sys
import time
import tracemalloc
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (RESULTS_JSON, EXPERIMENT_CONFIG, ALGO_NODE_CAP,
                    TIME_BUDGET_SEC, GLOBAL_SEED, ensure_dirs)
import generators as gen
from data_loader import load_dimacs, sample_connected_subgraph, ensure_dimacs_ny
from algorithms import dijkstra_heap, dijkstra_array, bellman_ford, floyd_warshall

SSSP_FUNCS = {
    "dijkstra_heap": dijkstra_heap,
    "dijkstra_array": dijkstra_array,
    "bellman_ford": bellman_ford,
}
ALL_ALGOS = ["dijkstra_heap", "dijkstra_array", "bellman_ford", "floyd_warshall"]


# ========================================================================== #
# Olcum yardimcilari                                                          #
# ========================================================================== #
def prepare_representation(algo_name, graph):
    """
    Bir algoritmanin ihtiyac duydugu gosterimi OLCUM DISINDA onceden kurar.
    Boylece zaman olcumu yalnizca algoritmanin kendisini, bellek olcumu de
    yalnizca algoritmanin KENDI calisma yapilarini (gosterim haric) yansitir.
    """
    if algo_name in ("dijkstra_array", "floyd_warshall"):
        graph.adjacency_matrix()
    elif algo_name == "dijkstra_heap":
        graph.adjacency_list()
    elif algo_name == "bellman_ford":
        graph.edge_list()


def measure_time(fn, repeats, warmup=True):
    """
    fn'i repeats kez calistirir; (medyan, en_kucuk) sureyi saniye doner.
    warmup=True ise olculmeyen bir isinma kosusu yapilir: ilk cagrinin
    onbellek/baslatma maliyetleri olcumu bozmaz.
    """
    if warmup:
        fn()
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times)


def measure_peak_memory(fn):
    """fn'i bir kez tracemalloc altinda calistirir; tepe bellegi (bayt) doner."""
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def trials_for(n, cfg, m=None):
    """
    Girdi buyudukce olcum tekrarini azaltir (toplam benchmark suresini
    sinirli tutmak icin). Hem dugum sayisi n hem de kenar sayisi m dikkate
    alinir; cunku bazi algoritmalarin maliyeti m'e baglidir (orn. Dijkstra
    O(E log V), Bellman-Ford O(V*E)).
    """
    S, R = cfg["sources"], cfg["repeats"]
    work = n if m is None else max(n, m // 4)
    if work > 6000:
        return 1, 1
    if work > 1500:
        return min(S, 2), 1
    return S, R


def pick_sources(n, count, seed):
    """Tekrarlanabilir sekilde `count` adet rastgele kaynak dugum secer."""
    rng = random.Random(seed * 7919 + 1)
    if count >= n:
        return list(range(n))
    return rng.sample(range(n), count)


def run_sssp(algo_name, graph, sources, repeats):
    """Tek-kaynakli bir algoritmayi olcer."""
    fn = SSSP_FUNCS[algo_name]
    prepare_representation(algo_name, graph)
    fn(graph, sources[0])  # (algoritma, graf) icin tek seferlik isinma kosusu
    per_source = []
    for s in sources:
        med, _ = measure_time(partial(fn, graph, s), repeats, warmup=False)
        per_source.append(med)
    time_per_query = statistics.median(per_source)
    peak = measure_peak_memory(partial(fn, graph, sources[0]))
    res = fn(graph, sources[0])  # sayaclar icin tek kosu
    return {
        "time_per_query": time_per_query,
        "peak_memory_bytes": peak,
        "relaxations": int(res.relaxations),
        "pops": int(res.pops),
        "passes": int(res.passes),
    }


def run_fw(graph, repeats):
    """Floyd-Warshall'i olcer. Tek bir sorgu = tum-ciftler on-islemesi."""
    prepare_representation("floyd_warshall", graph)
    # FW yavas ve tutarli oldugu icin pahali isinma kosusu yapilmaz.
    med, _ = measure_time(partial(floyd_warshall, graph), repeats, warmup=False)
    peak = measure_peak_memory(partial(floyd_warshall, graph))
    return {
        "time_per_query": med,        # tek sorgu icin bile tum O(V^3) gerekir
        "preprocess_time": med,
        "peak_memory_bytes": peak,
        "relaxations": graph.n ** 3,
        "pops": 0,
        "passes": 0,
    }


def skipped_record(experiment, n, m, algo, reason, extra=None):
    rec = {"experiment": experiment, "n": n, "m": m, "algorithm": algo,
           "feasible": False, "note": reason, "time_per_query": None,
           "peak_memory_bytes": None}
    if extra:
        rec.update(extra)
    return rec


# ========================================================================== #
# [A] Olceklenme deneyi                                                       #
# ========================================================================== #
def experiment_scaling(profile, cfg, seed):
    print("\n[A] OLCEKLENME DENEYI -- seyrek graflarda V'ye gore calisma suresi")
    sizes = cfg["scaling_sizes"]
    caps = ALGO_NODE_CAP[profile]
    stopped = set()       # zaman/bellek butcesi asilan algoritmalar
    records = []

    for n in sizes:
        g = gen.random_sparse(n, avg_degree=cfg["scaling_avg_degree"], seed=seed)
        rep_mem = g.memory_report()
        S, R = trials_for(n, cfg, g.m)
        sources = pick_sources(g.n, S, seed)
        print(f"  n={n:6d}  m={g.m:8d}  (yogunluk={g.density():.5f})")

        for algo in ALL_ALGOS:
            if algo in stopped:
                records.append(skipped_record("scaling", n, g.m, algo,
                               "daha kucuk boyutta uygulanabilirlik siniri asildi"))
                print(f"    {algo:16s}: ATLANDI (daha kucuk boyutta sinir asildi)")
                continue
            if n > caps[algo]:
                stopped.add(algo)
                records.append(skipped_record("scaling", n, g.m, algo,
                               f"karmasiklik/bellek siniri (n>{caps[algo]})"))
                print(f"    {algo:16s}: ATLANDI (n>{caps[algo]} -- teorik sinir)")
                continue

            if algo == "floyd_warshall":
                m = run_fw(g, R)
            else:
                m = run_sssp(algo, g, sources, R)

            rec = {"experiment": "scaling", "n": n, "m": g.m,
                   "graph_kind": "sparse", "algorithm": algo,
                   "feasible": True, "note": "", "rep_memory": rep_mem}
            rec.update(m)
            records.append(rec)
            print(f"    {algo:16s}: {m['time_per_query']*1000:10.3f} ms/sorgu"
                  f"   tepe bellek {m['peak_memory_bytes']/1024:9.1f} KB")

            if m["time_per_query"] > TIME_BUDGET_SEC:
                stopped.add(algo)
                print(f"      -> {algo} zaman butcesini ({TIME_BUDGET_SEC}s) asti; "
                      f"daha buyuk n'de calistirilmayacak")
    return records


# ========================================================================== #
# [B] Yogunluk deneyi                                                         #
# ========================================================================== #
def experiment_density(profile, cfg, seed):
    print("\n[B] YOGUNLUK DENEYI -- V sabit, kenar yogunlugu degisiyor")
    n = cfg["density_n"]
    caps = ALGO_NODE_CAP[profile]
    records = []

    for frac in cfg["density_fracs"]:
        g = gen.random_dense(n, edge_frac=frac, seed=seed)
        S, R = trials_for(n, cfg, g.m)
        sources = pick_sources(g.n, S, seed)
        print(f"  hedef yogunluk={frac:.2f}  n={n}  m={g.m:7d}  "
              f"gercek yogunluk={g.density():.4f}  ort.derece={g.m/g.n:.1f}")

        for algo in ALL_ALGOS:
            if n > caps[algo]:
                records.append(skipped_record("density", n, g.m, algo,
                               f"n>{caps[algo]}", extra={"edge_frac": frac,
                               "density": g.density()}))
                continue
            if algo == "floyd_warshall":
                m = run_fw(g, R)
            else:
                m = run_sssp(algo, g, sources, R)
            rec = {"experiment": "density", "n": n, "m": g.m,
                   "edge_frac": frac, "density": g.density(),
                   "algorithm": algo, "feasible": True, "note": ""}
            rec.update(m)
            records.append(rec)
            print(f"    {algo:16s}: {m['time_per_query']*1000:10.3f} ms/sorgu")
    return records


# ========================================================================== #
# [C] Sorgu sayisi / on-isleme odunlesimi                                     #
# ========================================================================== #
def experiment_many_queries(profile, cfg, seed):
    print("\n[C] SORGU SAYISI DENEYI -- on-isleme vs sorgu basina maliyet")
    n = cfg["manyq_n"]
    g = gen.random_sparse(n, avg_degree=cfg["manyq_avg_degree"], seed=seed)
    _, R = cfg["sources"], cfg["repeats"]
    sources = pick_sources(g.n, max(cfg["sources"], 8), seed)
    print(f"  sabit graf: n={n}  m={g.m}")

    # --- birim maliyetler ---
    prepare_representation("dijkstra_heap", g)
    dh_unit = statistics.median(
        measure_time(partial(dijkstra_heap, g, s), R)[0] for s in sources)

    prepare_representation("dijkstra_array", g)
    da_unit = statistics.median(
        measure_time(partial(dijkstra_array, g, s), R)[0] for s in sources)

    prepare_representation("floyd_warshall", g)
    fw_pre, _ = measure_time(partial(floyd_warshall, g), R)
    fw = floyd_warshall(g)

    # Floyd-Warshall'da bir sorgu = bir matris erisimi. 200 bin rastgele
    # erisimin ortalamasiyla birim sorgu maliyetini olc.
    rr = random.Random(seed * 13 + 7)
    pairs = [(rr.randrange(n), rr.randrange(n)) for _ in range(200_000)]
    gc.collect()
    t0 = time.perf_counter()
    acc = 0.0
    D = fw.dist
    for (a, b) in pairs:
        acc += D[a][b]
    fw_lookup = (time.perf_counter() - t0) / len(pairs)

    print(f"  Dijkstra-yigin  birim sorgu : {dh_unit*1e3:.4f} ms")
    print(f"  Dijkstra-dizi   birim sorgu : {da_unit*1e3:.4f} ms")
    print(f"  Floyd-Warshall  on-isleme   : {fw_pre*1e3:.2f} ms")
    print(f"  Floyd-Warshall  birim sorgu : {fw_lookup*1e9:.2f} ns (matris erisimi)")

    # --- Q sorgu icin toplam sureler ---
    totals = []
    for Q in cfg["manyq_counts"]:
        totals.append({
            "Q": Q,
            "dijkstra_heap": dh_unit * Q,
            "dijkstra_array": da_unit * Q,
            "floyd_warshall": fw_pre + fw_lookup * Q,
        })

    # Dijkstra-yigin ile Floyd-Warshall'in basabas (break-even) sorgu sayisi:
    #   fw_pre + fw_lookup*Q = dh_unit*Q   ->   Q* = fw_pre / (dh_unit - fw_lookup)
    breakeven = (fw_pre / (dh_unit - fw_lookup)
                 if dh_unit > fw_lookup else None)
    if breakeven:
        print(f"  Basabas noktasi (Dijkstra-yigin vs Floyd-Warshall): "
              f"Q* ~= {breakeven:.0f} sorgu")

    return {
        "n": n, "m": g.m,
        "dijkstra_heap_unit": dh_unit,
        "dijkstra_array_unit": da_unit,
        "floyd_warshall_preprocess": fw_pre,
        "floyd_warshall_lookup_unit": fw_lookup,
        "counts": cfg["manyq_counts"],
        "totals": totals,
        "breakeven_Q": breakeven,
    }


# ========================================================================== #
# [D] Gercek yol agi deneyi                                                   #
# ========================================================================== #
def experiment_real(profile, cfg, seed):
    print("\n[D] GERCEK YOL AGI DENEYI -- DIMACS New York")
    ensure_dimacs_ny()
    print("  Tam graf yukleniyor (DIMACS .gr)...")
    t0 = time.perf_counter()
    full = load_dimacs()
    print(f"  Yuklendi ({time.perf_counter()-t0:.1f}s): {full}")

    caps = ALGO_NODE_CAP[profile]
    subgraph_records = []

    # --- D1: kontrollu boyutta bagli alt-graflar ---
    for k in cfg["real_subgraph_sizes"]:
        sub = sample_connected_subgraph(full, k, seed=seed)
        rep_mem = sub.memory_report()
        S, R = trials_for(sub.n, cfg, sub.m)
        sources = pick_sources(sub.n, S, seed)
        print(f"  alt-graf  n={sub.n:5d}  m={sub.m:7d}  "
              f"(ort.derece={sub.m/sub.n:.2f})")
        for algo in ALL_ALGOS:
            if sub.n > caps[algo]:
                subgraph_records.append(skipped_record(
                    "real_subgraph", sub.n, sub.m, algo, f"n>{caps[algo]}"))
                print(f"    {algo:16s}: ATLANDI (n>{caps[algo]})")
                continue
            if algo == "floyd_warshall":
                m = run_fw(sub, R)
            else:
                m = run_sssp(algo, sub, sources, R)
            rec = {"experiment": "real_subgraph", "n": sub.n, "m": sub.m,
                   "graph_kind": "real_road", "algorithm": algo,
                   "feasible": True, "note": "", "rep_memory": rep_mem}
            rec.update(m)
            subgraph_records.append(rec)
            print(f"    {algo:16s}: {m['time_per_query']*1000:10.3f} ms/sorgu"
                  f"   tepe bellek {m['peak_memory_bytes']/1024:9.1f} KB")

    # --- D2: TAM graf (yalnizca Dijkstra-yigin uygulanabilir) ---
    print(f"  TAM GRAF  n={full.n}  m={full.m}:")
    full.adjacency_list()
    qn = cfg["real_full_queries"]
    sources = pick_sources(full.n, qn, seed)
    dijkstra_heap(full, sources[0])  # tek seferlik isinma kosusu
    full_sssp = statistics.median(
        measure_time(partial(dijkstra_heap, full, s), 1, warmup=False)[0]
        for s in sources)

    rr = random.Random(seed * 17 + 3)
    st_pairs = [(rr.randrange(full.n), rr.randrange(full.n)) for _ in range(qn)]
    early = statistics.median(
        measure_time(partial(dijkstra_heap, full, s, t), 1, warmup=False)[0]
        for (s, t) in st_pairs)

    peak = measure_peak_memory(partial(dijkstra_heap, full, sources[0]))
    rep_mem = full.memory_report()

    matrix_gb = full.n * full.n * 8 / 1e9
    bf_ops = full.n * full.m
    full_record = {
        "experiment": "real_full", "n": full.n, "m": full.m,
        "dijkstra_heap_full_sssp": full_sssp,
        "dijkstra_heap_early_stop": early,
        "peak_memory_bytes": peak,
        "rep_memory": rep_mem,
        "matrix_bytes_needed": full.n * full.n * 8,
        "infeasible": {
            "dijkstra_array": (f"komsuluk matrisi ~{matrix_gb:.0f} GB "
                               f"gerektirir -- bellege sigmaz"),
            "floyd_warshall": (f"O(V^2)~{matrix_gb:.0f} GB bellek ve "
                               f"O(V^3)~{full.n**3:.1e} islem -- uygulanamaz"),
            "bellman_ford": (f"O(V*E)~{bf_ops:.1e} islem/sorgu -- "
                             f"pratik degil"),
        },
    }
    print(f"    dijkstra_heap tam SSSP        : {full_sssp*1000:8.1f} ms/sorgu")
    print(f"    dijkstra_heap erken-durma s-t : {early*1000:8.1f} ms/sorgu")
    print(f"    tepe bellek                  : {peak/1e6:8.1f} MB")
    print(f"    -> Dijkstra-dizi & Floyd-Warshall: komsuluk matrisi "
          f"~{matrix_gb:.0f} GB -- UYGULANAMAZ")

    return {"subgraphs": subgraph_records, "full": full_record}


# ========================================================================== #
# Ana akis                                                                    #
# ========================================================================== #
def main():
    parser = argparse.ArgumentParser(description="En-kisa-yol benchmark kosucusu")
    parser.add_argument("--full", action="store_true",
                        help="tam deney profili (rapor sonuclari)")
    parser.add_argument("--quick", action="store_true",
                        help="hizli profil (gelistirme/demo) -- varsayilan")
    args = parser.parse_args()
    profile = "full" if args.full else "quick"

    ensure_dirs()
    cfg = EXPERIMENT_CONFIG[profile]
    seed = GLOBAL_SEED

    print("=" * 70)
    print(f"  BENCHMARK  --  profil: {profile.upper()}")
    print("=" * 70)
    t_start = time.perf_counter()

    results = {
        "meta": {
            "profile": profile,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "python": sys.version.split()[0],
            "config": cfg,
            "time_budget_sec": TIME_BUDGET_SEC,
            "algo_node_cap": ALGO_NODE_CAP[profile],
        },
        "scaling": experiment_scaling(profile, cfg, seed),
        "density": experiment_density(profile, cfg, seed),
        "many_queries": experiment_many_queries(profile, cfg, seed),
        "real": experiment_real(profile, cfg, seed),
    }

    elapsed = time.perf_counter() - t_start
    results["meta"]["elapsed_sec"] = elapsed

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  BENCHMARK TAMAMLANDI -- toplam sure {elapsed:.1f}s")
    print(f"  Sonuclar: {RESULTS_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()

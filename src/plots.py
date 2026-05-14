# -*- coding: utf-8 -*-
"""
plots.py -- Benchmark sonuclarindan grafik ve tablo uretir.

Girdi : results/raw/benchmark_results.json   (benchmark.py uretir)
Cikti : results/plots/*.png                  (rapor ve sunum icin)
        results/tables/*.csv, *.md           (sayisal tablolar)

Calistirma:  python src/plots.py             # Turkce (varsayilan)
             python src/plots.py --lang en   # Ingilizce -> results/plots_en/

Ingilizce secildiginde ciktilar ayri klasorlere (plots_en/, tables_en/)
yazilir; Turkce ciktilarin uzerine yazilmaz.
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (RESULTS_JSON, PLOTS_DIR, TABLES_DIR, PLOTS_DIR_EN,
                    TABLES_DIR_EN, ensure_dirs, ALGO_SHORT, ALGO_SHORT_EN,
                    ALGO_COLORS)

ALGO_MARKERS = {
    "dijkstra_heap": "o",
    "dijkstra_array": "s",
    "bellman_ford": "^",
    "floyd_warshall": "D",
}
ALL_ALGOS = ["dijkstra_heap", "dijkstra_array", "bellman_ford", "floyd_warshall"]

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


# -------------------------------------------------------------------------- #
# Dil (i18n) -- tum kullaniciya gorunen metinler buradan gelir                #
# -------------------------------------------------------------------------- #
LANG = "tr"   # main() icinde --lang ile degistirilir

STRINGS = {
    "tr": {
        "written": "  yazildi: {path}",
        "written2": "  yazildi: {csv} , {md}",
        "err_no_results": "HATA: {path} bulunamadi. "
                          "Once: python src/benchmark.py --full",
        "loaded": "Sonuclar yuklendi (profil: {profile}).",
        "gen_plots": "Grafikler uretiliyor...",
        "gen_tables": "Tablolar uretiliyor...",
        "done": "Tamamlandi.",
        # eksen etiketleri
        "xl_nodes": "Dugum sayisi  V  (log olcek)",
        "yl_time_ms": "Sorgu basina sure  (ms, log olcek)",
        "yl_relax": "Gevsetme (relaxation) sayisi  (log olcek)",
        "xl_avgdeg": "Ortalama derece  (m / V)  --  artan yogunluk",
        "xl_queries": "Sorgu sayisi  Q  (log olcek)",
        "yl_total_s": "Toplam sure  (s, log olcek)",
        "yl_mem_kb": "Bellek ayak izi  (KB, log olcek)",
        "yl_algmem_kb": "Algoritmanin tepe calisma bellegi  (KB, log olcek)",
        "xl_sub_nodes": "Alt-graf dugum sayisi  V  (log olcek)",
        "yl_time_ms_plain": "Sorgu basina sure  (ms)",
        # baslıklar
        "t_scaling_runtime": "Deney A -- Seyrek graflarda olceklenme\n"
                             "(parantez icinde ampirik us; teorik usse "
                             "yakinligina dikkat)",
        "t_scaling_ops": "Deney A -- Yapilan is miktari (gevsetme sayisi)\n"
                         "Calisma suresinden bagimsiz, donanimdan bagimsiz "
                         "olcum",
        "t_density": "Deney B -- Yogunlugun etkisi  (V = {n} sabit)\n"
                     "Dijkstra-Dizi ve Floyd-Warshall yogunluktan bagimsiz; "
                     "Dijkstra-Yigin ve Bellman-Ford yogunlukla artar",
        "t_manyq": "Deney C -- On-isleme vs sorgu odunlesimi  (V = {n})\n"
                   "Floyd-Warshall: yuksek sabit on-isleme, ~bedava sorgu. "
                   "Q > Q* icin Floyd-Warshall karli",
        "anno_breakeven": "Basabas Q* ~= {be}",
        "t_mem_rep": "Graf gosterimlerinin bellek ayak izi\n"
                     "Komsuluk matrisi O(V^2) ile hizla buyur; seyrek "
                     "graflarda kenar/komsuluk listesi cok daha ekonomik",
        "t_mem_alg": "Algoritmalarin calisma bellegi (gosterim haric)\n"
                     "Floyd-Warshall O(V^2); tek-kaynakli algoritmalar "
                     "O(V) veya O(V+E)",
        "rep_edge": "Kenar listesi  O(E)",
        "rep_adjl": "Komsuluk listesi  O(V+E)",
        "rep_adjm": "Komsuluk matrisi  O(V^2)",
        "t_real_sub": "New York yol agindan ornek alt-graflar",
        "t_real_full": "Tam graf  (V = {n}, E = {m})",
        "real_suptitle": "Deney D -- Gercek yol agi (DIMACS New York)",
        "bar_dh_full": "Dijkstra-Yigin\n(tam SSSP)",
        "bar_dh_early": "Dijkstra-Yigin\n(erken durma)",
        "bar_da": "Dijkstra-Dizi",
        "bar_fw": "Floyd-Warshall",
        "bar_bf": "Bellman-Ford",
        "infeasible_mark": "UYGULANAMAZ",
        "matrix_note": "Komsuluk matrisi ~{gb} GB gerektirir\n"
                       "(matris tabanli yaklasimlar bellege sigmaz)",
        # tablo basliklari / hucreleri
        "infeasible": "uygulanamaz",
        "th_avgdeg": "Ort.derece (m/V)",
        "th_density": "Yogunluk",
        "th_q": "Q (sorgu sayisi)",
        "th_dh_s": "Dijkstra-Yigin (s)",
        "th_da_s": "Dijkstra-Dizi (s)",
        "th_fw_s": "Floyd-Warshall (s)",
        "th_v_sub": "V (alt-graf)",
    },
    "en": {
        "written": "  written: {path}",
        "written2": "  written: {csv} , {md}",
        "err_no_results": "ERROR: {path} not found. "
                          "First run: python src/benchmark.py --full",
        "loaded": "Results loaded (profile: {profile}).",
        "gen_plots": "Generating plots...",
        "gen_tables": "Generating tables...",
        "done": "Done.",
        # axis labels
        "xl_nodes": "Number of nodes  V  (log scale)",
        "yl_time_ms": "Time per query  (ms, log scale)",
        "yl_relax": "Number of relaxations  (log scale)",
        "xl_avgdeg": "Average degree  (m / V)  --  increasing density",
        "xl_queries": "Number of queries  Q  (log scale)",
        "yl_total_s": "Total time  (s, log scale)",
        "yl_mem_kb": "Memory footprint  (KB, log scale)",
        "yl_algmem_kb": "Algorithm peak working memory  (KB, log scale)",
        "xl_sub_nodes": "Subgraph node count  V  (log scale)",
        "yl_time_ms_plain": "Time per query  (ms)",
        # titles
        "t_scaling_runtime": "Experiment A -- Scaling on sparse graphs\n"
                             "(empirical exponent in parentheses; note how "
                             "close it is to the theoretical one)",
        "t_scaling_ops": "Experiment A -- Amount of work done (relaxation "
                         "count)\nIndependent of running time, independent "
                         "of hardware",
        "t_density": "Experiment B -- The effect of density  (V = {n} "
                     "fixed)\nDijkstra-Array and Floyd-Warshall are "
                     "independent of density; Dijkstra-Heap and "
                     "Bellman-Ford rise with it",
        "t_manyq": "Experiment C -- Preprocessing vs. query trade-off  "
                   "(V = {n})\nFloyd-Warshall: high fixed preprocessing, "
                   "~free queries. For Q > Q*, Floyd-Warshall is profitable",
        "anno_breakeven": "Break-even Q* ~= {be}",
        "t_mem_rep": "Memory footprint of the graph representations\n"
                     "The adjacency matrix grows fast as O(V^2); on sparse "
                     "graphs the edge/adjacency list is far more economical",
        "t_mem_alg": "Algorithms' working memory (excluding the "
                     "representation)\nFloyd-Warshall O(V^2); single-source "
                     "algorithms O(V) or O(V+E)",
        "rep_edge": "Edge list  O(E)",
        "rep_adjl": "Adjacency list  O(V+E)",
        "rep_adjm": "Adjacency matrix  O(V^2)",
        "t_real_sub": "Sample subgraphs from the New York road network",
        "t_real_full": "Full graph  (V = {n}, E = {m})",
        "real_suptitle": "Experiment D -- Real road network "
                         "(DIMACS New York)",
        "bar_dh_full": "Dijkstra-Heap\n(full SSSP)",
        "bar_dh_early": "Dijkstra-Heap\n(early stop)",
        "bar_da": "Dijkstra-Array",
        "bar_fw": "Floyd-Warshall",
        "bar_bf": "Bellman-Ford",
        "infeasible_mark": "INFEASIBLE",
        "matrix_note": "Adjacency matrix requires ~{gb} GB\n"
                       "(matrix-based approaches do not fit in memory)",
        # table headers / cells
        "infeasible": "infeasible",
        "th_avgdeg": "Avg. degree (m/V)",
        "th_density": "Density",
        "th_q": "Q (query count)",
        "th_dh_s": "Dijkstra-Heap (s)",
        "th_da_s": "Dijkstra-Array (s)",
        "th_fw_s": "Floyd-Warshall (s)",
        "th_v_sub": "V (subgraph)",
    },
}


def T(key):
    """Gecerli dildeki metni dondurur."""
    return STRINGS[LANG][key]


def _algo_short():
    """Gecerli dildeki algoritma kisa adlari sozlugu."""
    return ALGO_SHORT_EN if LANG == "en" else ALGO_SHORT


def _plots_dir():
    return PLOTS_DIR_EN if LANG == "en" else PLOTS_DIR


def _tables_dir():
    return TABLES_DIR_EN if LANG == "en" else TABLES_DIR


def _parse_lang(argv):
    """--lang en / --lang=en  ya da  PROJECT_LANG ortam degiskeni."""
    lang = os.environ.get("PROJECT_LANG", "tr")
    for i, a in enumerate(argv):
        if a == "--lang" and i + 1 < len(argv):
            lang = argv[i + 1]
        elif a.startswith("--lang="):
            lang = a.split("=", 1)[1]
    return lang if lang in ("tr", "en") else "tr"


# -------------------------------------------------------------------------- #
# Yardimcilar                                                                 #
# -------------------------------------------------------------------------- #
def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)


def series(records, algorithm, xkey, ykey):
    """Belirli bir algoritma icin (x, y) noktalarini siralanmis dondurur."""
    pts = []
    for r in records:
        if r.get("algorithm") != algorithm:
            continue
        if not r.get("feasible"):
            continue
        x, y = r.get(xkey), r.get(ykey)
        if x is None or y is None:
            continue
        pts.append((x, y))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def power_law_exponent(xs, ys):
    """
    log-log uzayda dogru uydurarak ampirik us (exponent) tahmin eder.

    Uydurma, dizinin ASIMPTOTIK KUYRUGU uzerinde yapilir (en buyuk
    girdiler). Cunku kucuk girdilerde sabit carpanlar (NumPy cagri yuku,
    yigin kurulumu) baskindir ve log-log egimini gercek asimptotik
    davranisin altina ceker. Buyuk girdilere odaklanmak teorik usse
    daha sadik bir tahmin verir.
    """
    if len(xs) < 2:
        return None
    import math
    k = max(3, math.ceil(0.55 * len(xs)))
    k = min(k, len(xs))
    lx = np.log(np.array(xs[-k:], dtype=float))
    ly = np.log(np.array(ys[-k:], dtype=float))
    slope, _ = np.polyfit(lx, ly, 1)
    return slope


def _save(fig, name):
    path = os.path.join(_plots_dir(), name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(T("written").format(path=path))


# -------------------------------------------------------------------------- #
# [A] Olceklenme -- calisma suresi                                            #
# -------------------------------------------------------------------------- #
def plot_scaling_runtime(results):
    recs = results["scaling"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for algo in ALL_ALGOS:
        xs, ys = series(recs, algo, "n", "time_per_query")
        if not xs:
            continue
        ys_ms = [y * 1000 for y in ys]
        exp = power_law_exponent(xs, ys)
        label = _algo_short()[algo]
        if exp is not None:
            label += f"  (~V^{exp:.2f})"
        ax.plot(xs, ys_ms, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                label=label, linewidth=1.8, markersize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(T("xl_nodes"))
    ax.set_ylabel(T("yl_time_ms"))
    ax.set_title(T("t_scaling_runtime"))
    ax.legend(framealpha=0.95)
    _save(fig, "A_scaling_runtime.png")


def plot_scaling_operations(results):
    recs = results["scaling"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for algo in ALL_ALGOS:
        xs, ys = series(recs, algo, "n", "relaxations")
        if not xs:
            continue
        exp = power_law_exponent(xs, ys)
        label = _algo_short()[algo]
        if exp is not None:
            label += f"  (~V^{exp:.2f})"
        ax.plot(xs, ys, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                label=label, linewidth=1.8, markersize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(T("xl_nodes"))
    ax.set_ylabel(T("yl_relax"))
    ax.set_title(T("t_scaling_ops"))
    ax.legend(framealpha=0.95)
    _save(fig, "A_scaling_operations.png")


# -------------------------------------------------------------------------- #
# [B] Yogunluk                                                                #
# -------------------------------------------------------------------------- #
def plot_density(results):
    recs = results["density"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    n_val = recs[0]["n"] if recs else "?"
    for algo in ALL_ALGOS:
        # x ekseni: ortalama derece (m / n) -- yogunlugun okunabilir olcusu
        pts = []
        for r in recs:
            if r.get("algorithm") == algo and r.get("feasible"):
                pts.append((r["m"] / r["n"], r["time_per_query"] * 1000))
        pts.sort()
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                label=_algo_short()[algo], linewidth=1.8, markersize=6)
    ax.set_yscale("log")
    ax.set_xlabel(T("xl_avgdeg"))
    ax.set_ylabel(T("yl_time_ms"))
    ax.set_title(T("t_density").format(n=n_val))
    ax.legend(framealpha=0.95)
    _save(fig, "B_density_runtime.png")


# -------------------------------------------------------------------------- #
# [C] Sorgu sayisi / on-isleme odunlesimi                                     #
# -------------------------------------------------------------------------- #
def plot_many_queries(results):
    mq = results["many_queries"]
    totals = mq["totals"]
    counts = [t["Q"] for t in totals]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for algo in ["dijkstra_heap", "dijkstra_array", "floyd_warshall"]:
        ys = [t[algo] for t in totals]
        ax.plot(counts, ys, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                label=_algo_short()[algo], linewidth=1.8, markersize=6)
    be = mq.get("breakeven_Q")
    if be:
        ax.axvline(be, color="gray", linestyle="--", linewidth=1.2)
        ax.annotate(T("anno_breakeven").format(be=f"{be:.0f}"),
                    xy=(be, ax.get_ylim()[0]),
                    xytext=(be * 1.1, max(t["dijkstra_heap"] for t in totals) * 0.05),
                    color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(T("xl_queries"))
    ax.set_ylabel(T("yl_total_s"))
    ax.set_title(T("t_manyq").format(n=mq["n"]))
    ax.legend(framealpha=0.95)
    _save(fig, "C_many_queries.png")


# -------------------------------------------------------------------------- #
# Bellek -- gosterimler                                                       #
# -------------------------------------------------------------------------- #
def plot_memory_representations(results):
    recs = results["scaling"]
    # rep_memory her n icin (algoritmadan bagimsiz) -- dijkstra_heap'ten al
    pts = {}
    for r in recs:
        if r.get("algorithm") == "dijkstra_heap" and r.get("rep_memory"):
            pts[r["n"]] = r["rep_memory"]
    ns = sorted(pts)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    styles = {
        "edge_list": (T("rep_edge"), "#2ca02c", "^"),
        "adjacency_list": (T("rep_adjl"), "#1f77b4", "o"),
        "adjacency_matrix": (T("rep_adjm"), "#d62728", "s"),
    }
    for key, (label, color, marker) in styles.items():
        ys = [pts[n][key] / 1024 for n in ns]   # KB
        ax.plot(ns, ys, marker=marker, color=color, label=label,
                linewidth=1.8, markersize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(T("xl_nodes"))
    ax.set_ylabel(T("yl_mem_kb"))
    ax.set_title(T("t_mem_rep"))
    ax.legend(framealpha=0.95)
    _save(fig, "D_memory_representations.png")


def plot_memory_algorithms(results):
    recs = results["scaling"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for algo in ALL_ALGOS:
        xs, ys = series(recs, algo, "n", "peak_memory_bytes")
        if not xs:
            continue
        ys_kb = [y / 1024 for y in ys]
        exp = power_law_exponent(xs, ys)
        label = _algo_short()[algo]
        if exp is not None:
            label += f"  (~V^{exp:.2f})"
        ax.plot(xs, ys_kb, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                label=label, linewidth=1.8, markersize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(T("xl_nodes"))
    ax.set_ylabel(T("yl_algmem_kb"))
    ax.set_title(T("t_mem_alg"))
    ax.legend(framealpha=0.95)
    _save(fig, "D_memory_algorithms.png")


# -------------------------------------------------------------------------- #
# [D] Gercek yol agi                                                          #
# -------------------------------------------------------------------------- #
def plot_real(results):
    real = results["real"]
    subs = real["subgraphs"]
    full = real["full"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Sol panel: alt-graf olceklenmesi (dort algoritma)
    for algo in ALL_ALGOS:
        xs, ys = series(subs, algo, "n", "time_per_query")
        if not xs:
            continue
        ys_ms = [y * 1000 for y in ys]
        ax1.plot(xs, ys_ms, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                 label=_algo_short()[algo], linewidth=1.8, markersize=6)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(T("xl_sub_nodes"))
    ax1.set_ylabel(T("yl_time_ms"))
    ax1.set_title(T("t_real_sub"))
    ax1.legend(framealpha=0.95)

    # Sag panel: tam graf -- uygulanabilir vs uygulanamaz
    labels = [T("bar_dh_full"), T("bar_dh_early"), T("bar_da"),
              T("bar_fw"), T("bar_bf")]
    full_sssp = full["dijkstra_heap_full_sssp"] * 1000
    early = full["dijkstra_heap_early_stop"] * 1000
    vals = [full_sssp, early, 0, 0, 0]
    colors = [ALGO_COLORS["dijkstra_heap"], ALGO_COLORS["dijkstra_heap"],
              ALGO_COLORS["dijkstra_array"], ALGO_COLORS["floyd_warshall"],
              ALGO_COLORS["bellman_ford"]]
    bars = ax2.bar(labels, vals, color=colors, alpha=0.85)
    ax2.set_ylabel(T("yl_time_ms_plain"))
    ax2.set_title(T("t_real_full").format(n=f"{full['n']:,}",
                                          m=f"{full['m']:,}"))
    # uygulanamaz olanlari isaretle
    matrix_gb = full["matrix_bytes_needed"] / 1e9
    for i, name in [(2, "dijkstra_array"), (3, "floyd_warshall"),
                    (4, "bellman_ford")]:
        ax2.text(i, full_sssp * 0.04,
                 T("infeasible_mark"), ha="center", va="bottom", rotation=90,
                 color="darkred", fontweight="bold", fontsize=9)
    ax2.text(0.5, 0.92,
             T("matrix_note").format(gb=f"{matrix_gb:.0f}"),
             transform=ax2.transAxes, ha="center", va="top", color="darkred",
             fontsize=9, bbox=dict(boxstyle="round", fc="mistyrose", ec="darkred"))
    fig.suptitle(T("real_suptitle"), y=1.02, fontsize=12, fontweight="bold")
    _save(fig, "E_real_road_network.png")


# -------------------------------------------------------------------------- #
# Tablolar (CSV + Markdown)                                                   #
# -------------------------------------------------------------------------- #
def _write_table(name, header, rows):
    csv_path = os.path.join(_tables_dir(), name + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md_path = os.path.join(_tables_dir(), name + ".md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")
    print(T("written2").format(csv=csv_path, md=md_path))


def write_tables(results):
    short = _algo_short()
    infeasible = T("infeasible")
    # Tablo 1: olceklenme (ms/sorgu)
    recs = results["scaling"]
    ns = sorted({r["n"] for r in recs})
    header = ["V"] + [short[a] for a in ALL_ALGOS]
    rows = []
    for n in ns:
        row = [n]
        for a in ALL_ALGOS:
            rec = next((r for r in recs if r["n"] == n and r["algorithm"] == a),
                       None)
            if rec and rec.get("feasible"):
                row.append(f"{rec['time_per_query']*1000:.3f}")
            else:
                row.append(infeasible)
        rows.append(row)
    _write_table("A_scaling_runtime_ms", header, rows)

    # Tablo 2: yogunluk (ms/sorgu)
    drecs = results["density"]
    fracs = sorted({r["edge_frac"] for r in drecs})
    header = [T("th_avgdeg"), T("th_density")] + [short[a] for a in ALL_ALGOS]
    rows = []
    for fr in fracs:
        sample = next(r for r in drecs if r["edge_frac"] == fr)
        row = [f"{sample['m']/sample['n']:.1f}", f"{sample['density']:.4f}"]
        for a in ALL_ALGOS:
            rec = next((r for r in drecs
                        if r["edge_frac"] == fr and r["algorithm"] == a), None)
            if rec and rec.get("feasible"):
                row.append(f"{rec['time_per_query']*1000:.3f}")
            else:
                row.append(infeasible)
        rows.append(row)
    _write_table("B_density_runtime_ms", header, rows)

    # Tablo 3: sorgu sayisi (toplam sure, s)
    mq = results["many_queries"]
    header = [T("th_q"), T("th_dh_s"), T("th_da_s"), T("th_fw_s")]
    rows = []
    for t in mq["totals"]:
        rows.append([t["Q"], f"{t['dijkstra_heap']:.4f}",
                     f"{t['dijkstra_array']:.4f}", f"{t['floyd_warshall']:.4f}"])
    _write_table("C_many_queries_total_sec", header, rows)

    # Tablo 4: gercek alt-graflar
    subs = results["real"]["subgraphs"]
    ns = sorted({r["n"] for r in subs})
    header = [T("th_v_sub")] + [short[a] for a in ALL_ALGOS]
    rows = []
    for n in ns:
        row = [n]
        for a in ALL_ALGOS:
            rec = next((r for r in subs
                        if r["n"] == n and r["algorithm"] == a), None)
            if rec and rec.get("feasible"):
                row.append(f"{rec['time_per_query']*1000:.3f}")
            else:
                row.append(infeasible)
        rows.append(row)
    _write_table("D_real_subgraph_runtime_ms", header, rows)


# -------------------------------------------------------------------------- #
# Ana akis                                                                    #
# -------------------------------------------------------------------------- #
def main():
    global LANG
    LANG = _parse_lang(sys.argv[1:])
    ensure_dirs()
    if not os.path.exists(RESULTS_JSON):
        print(T("err_no_results").format(path=RESULTS_JSON))
        return 1
    results = load_results()
    print(T("loaded").format(profile=results["meta"]["profile"]))
    print(T("gen_plots"))
    plot_scaling_runtime(results)
    plot_scaling_operations(results)
    plot_density(results)
    plot_many_queries(results)
    plot_memory_representations(results)
    plot_memory_algorithms(results)
    plot_real(results)
    print(T("gen_tables"))
    write_tables(results)
    print(T("done"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

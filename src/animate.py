# -*- coding: utf-8 -*-
"""
animate.py -- GORSEL SIMULASYON: algoritmalar grafi "kesfederken" izlenir.

Iki gorsel uretir:

  1) Izgara kesfi  (Dijkstra  vs  Bellman-Ford, yan yana)
     Ayni izgara graf uzerinde iki algoritmanin AYNI sorguyu nasil farkli
     bicimde cozdugu adim adim gosterilir:
       - Dijkstra : kaynaktan disa dogru, UZAKLIK SIRASIYLA dugum kesinlestirir;
                    yalnizca ince bir "sinir" (frontier) uzerinde calisir.
       - Bellman-Ford : her turda AYRIM YAPMADAN tum kenarlari gevsetir;
                    uzaklik alani disa dogru dalga gibi yayilir.

  2) Floyd-Warshall matrisi
     Uzaklik matrisi D'nin, ara dugum k arttikca nasil "dolduguunu" gosterir
     -- tek-kaynakli kesif yerine tum-ciftler dinamik programlama.

Her gorsel hem animasyonlu GIF hem de rapora/sunuma konabilen statik bir
"anlik goruntuler" (snapshots) PNG'si olarak kaydedilir.

Calistirma:  python src/animate.py             # Turkce (varsayilan)
             python src/animate.py --lang en   # Ingilizce -> results/*_en/
Cikti:       results/animations/*.gif  +  results/plots/F_*_snapshots.png
             (Ingilizce: results/animations_en/  +  results/plots_en/)
"""
from __future__ import annotations

import heapq
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np

from config import (INF, ANIM_DIR, PLOTS_DIR, ANIM_DIR_EN, PLOTS_DIR_EN,
                    GLOBAL_SEED, ensure_dirs)
import generators as gen

# Durum renkleri (Dijkstra paneli)
C_UNSEEN = "#ededed"     # gorulmedi
C_FRONTIER = "#ff9e3d"   # sinirda (yiginla, henuz kesinlesmedi)
C_DONE = "#1f6fb4"       # kesinlesti (uzakligi nihai)
C_PATH = "#d62728"       # en kisa yol

plt.rcParams.update({"font.size": 10})


# -------------------------------------------------------------------------- #
# Dil (i18n) -- tum kullaniciya gorunen metinler buradan gelir                #
# -------------------------------------------------------------------------- #
LANG = "tr"   # main() icinde --lang ile degistirilir

STRINGS = {
    "tr": {
        "gen": "Gorsel simulasyonlar uretiliyor...",
        "step1": "  [1/2] Izgara kesfi (Dijkstra vs Bellman-Ford)...",
        "step2": "  [2/2] Floyd-Warshall matris animasyonu...",
        "written": "      yazildi: {path}",
        "gif_fail": "      (GIF uretilemedi: {exc} -- snapshot PNG yeterli)",
        "done": "Tamamlandi.",
        # izgara kesfi -- gorsel ici metinler
        "dij_title": "Dijkstra  —  {step}/{total} dugum kesinlesti\n"
                     "(yalnizca ince 'sinir' uzerinde calisir)",
        "bf_title": "Bellman-Ford  —  tur {p}/{passes}\n"
                    "(her tur: AYRIMSIZ tum {edges} kenar gevsetilir)",
        "src_mark": "S",
        "tgt_mark": "H",
        "pct": "%{p}",
        "ylabel_dij": "Dijkstra",
        "ylabel_bf": "Bellman-Ford",
        "grid_suptitle": "Izgara kesfi: Dijkstra secerek ilerler, "
                         "Bellman-Ford tum kenarlari tarar  "
                         "(S=kaynak, H=hedef, kirmizi=en kisa yol)",
        # floyd-warshall -- gorsel ici metinler
        "fw_snap_title": "{k} ara dugum\nmatris %{pct} dolu",
        "fw_suptitle": "Floyd-Warshall: uzaklik matrisi D ({n}x{n}) ara "
                       "dugum sayisi arttikca dolar  (gri = henuz yol yok)",
        "fw_cbar": "en kisa uzaklik",
        "fw_gif_title": "Floyd-Warshall  —  {k} / {n} ara dugum kullanildi\n"
                        "matris %{pct} dolu",
    },
    "en": {
        "gen": "Generating visual simulations...",
        "step1": "  [1/2] Grid exploration (Dijkstra vs Bellman-Ford)...",
        "step2": "  [2/2] Floyd-Warshall matrix animation...",
        "written": "      written: {path}",
        "gif_fail": "      (could not produce GIF: {exc} -- snapshot PNG is "
                    "sufficient)",
        "done": "Done.",
        # grid exploration -- in-image text
        "dij_title": "Dijkstra  —  {step}/{total} nodes finalized\n"
                     "(works only on the thin 'frontier')",
        "bf_title": "Bellman-Ford  —  pass {p}/{passes}\n"
                    "(each pass: ALL {edges} edges relaxed INDISCRIMINATELY)",
        "src_mark": "S",
        "tgt_mark": "T",
        "pct": "{p}%",
        "ylabel_dij": "Dijkstra",
        "ylabel_bf": "Bellman-Ford",
        "grid_suptitle": "Grid exploration: Dijkstra advances selectively, "
                         "Bellman-Ford scans all edges  "
                         "(S=source, T=target, red=shortest path)",
        # floyd-warshall -- in-image text
        "fw_snap_title": "{k} intermediate nodes\nmatrix {pct}% filled",
        "fw_suptitle": "Floyd-Warshall: the distance matrix D ({n}x{n}) "
                       "fills in as the number of intermediate nodes grows  "
                       "(gray = no path yet)",
        "fw_cbar": "shortest distance",
        "fw_gif_title": "Floyd-Warshall  —  {k} / {n} intermediate nodes "
                        "used\nmatrix {pct}% filled",
    },
}


def T(key):
    """Gecerli dildeki metni dondurur."""
    return STRINGS[LANG][key]


def _plots_dir():
    return PLOTS_DIR_EN if LANG == "en" else PLOTS_DIR


def _anim_dir():
    return ANIM_DIR_EN if LANG == "en" else ANIM_DIR


def _parse_lang(argv):
    """--lang en / --lang=en  ya da  PROJECT_LANG ortam degiskeni."""
    lang = os.environ.get("PROJECT_LANG", "tr")
    for i, a in enumerate(argv):
        if a == "--lang" and i + 1 < len(argv):
            lang = argv[i + 1]
        elif a.startswith("--lang="):
            lang = a.split("=", 1)[1]
    return lang if lang in ("tr", "en") else "tr"


# ========================================================================== #
# Izlenen (instrumented) algoritmalar -- ana surumlerin aynisi, her adimin    #
# durumunu kaydeden ek defterlerle.                                           #
# ========================================================================== #
def trace_dijkstra(graph, source):
    """
    dijkstra_heap ile ayni mantik; ek olarak her dugum icin:
      fin_step[v]  = v'nin kac dugum kesinlestikten sonra kesinlestigi (-1: hic)
      seen_step[v] = v'nin ilk kez sonlu uzakliga ulastigi adim (-1: hic)
    """
    adj = graph.adjacency_list()
    n = graph.n
    dist = [INF] * n
    pred = [-1] * n
    visited = [False] * n
    fin_step = [-1] * n
    seen_step = [-1] * n

    dist[source] = 0.0
    seen_step[source] = 0
    heap = [(0.0, source)]
    step = 0
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        step += 1
        fin_step[u] = step
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                if dist[v] == INF:
                    seen_step[v] = step
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))
    return {"fin_step": np.array(fin_step), "seen_step": np.array(seen_step),
            "pred": pred, "dist": dist, "total": step}


def trace_bellman_ford(graph, source, edge_order="reverse"):
    """
    bellman_ford ile ayni mantik; her TURDAN sonra dist dizisinin bir
    anlik goruntusu saklanir.

    Animasyonda Bellman-Ford'un klasik 'dalga' davranisi (her turda
    yaklasik bir adim ilerleme) net gorulsun diye kenarlar TERS sirada
    islenir. Bu, BF icin en-kotu-durum kenar sirasidir: algoritma yine
    DOGRU sonucu verir, sadece daha cok tur yapar. (Ana benchmark dogal
    kenar sirasini kullanir.) Erken durma yine aciktir.
    """
    edges = list(graph.edge_list())
    if edge_order == "reverse":
        edges = edges[::-1]
    elif edge_order == "shuffle":
        random.Random(1).shuffle(edges)
    n = graph.n
    dist = [INF] * n
    pred = [-1] * n
    dist[source] = 0.0
    snapshots = [list(dist)]            # tur 0: yalnizca kaynak
    for _ in range(n - 1):
        changed = False
        for u, v, w in edges:
            du = dist[u]
            if du != INF and du + w < dist[v]:
                dist[v] = du + w
                pred[v] = u
                changed = True
        snapshots.append(list(dist))
        if not changed:
            break
    return {"snapshots": snapshots, "pred": pred,
            "passes": len(snapshots) - 1, "edges": len(edges)}


def trace_floyd_warshall(graph, max_frames=55):
    """
    floyd_warshall ile ayni mantik; her ara dugum k'dan sonra uzaklik
    matrisi D'nin bir kopyasi saklanir. Cok dugumlu graflarda kareler
    `max_frames`e indirgenir.
    """
    M = graph.adjacency_matrix()
    n = graph.n
    D = M.copy()
    snaps = [D.copy()]
    for k in range(n):
        through_k = D[:, k][:, None] + D[k, :][None, :]
        np.minimum(D, through_k, out=D)
        snaps.append(D.copy())
    labels = list(range(len(snaps)))   # i = kullanilan ara dugum sayisi
    if len(snaps) > max_frames:
        idx = np.linspace(0, len(snaps) - 1, max_frames).astype(int)
        snaps = [snaps[i] for i in idx]
        labels = [labels[i] for i in idx]
    return {"snapshots": snaps, "labels": labels, "n": n}


# ========================================================================== #
# Yardimcilar                                                                 #
# ========================================================================== #
def reconstruct(pred, source, target):
    path, v, guard = [], target, 0
    while v != -1 and guard <= len(pred):
        path.append(v)
        if v == source:
            return path[::-1]
        v = pred[v]
        guard += 1
    return []


def dijkstra_state(trace, rows, cols, step):
    """Dijkstra panelinin `step` adimindaki rows x cols durum dizisi
    (0=gorulmedi, 1=sinirda, 2=kesinlesti)."""
    fs, ss = trace["fin_step"], trace["seen_step"]
    state = np.zeros(rows * cols, dtype=int)
    state[(ss != -1) & (ss <= step)] = 1
    state[(fs != -1) & (fs <= step)] = 2
    return state.reshape(rows, cols)


def bf_field(trace, rows, cols, p):
    """Bellman-Ford panelinin `p`. turdaki uzaklik alani (rows x cols);
    erisilmeyen dugumler NaN."""
    snap = trace["snapshots"][p]
    arr = np.array([np.nan if d == INF else d for d in snap], dtype=float)
    return arr.reshape(rows, cols)


def _draw_path(ax, path, cols, color=C_PATH):
    if not path:
        return
    xs = [v % cols for v in path]
    ys = [v // cols for v in path]
    ax.plot(xs, ys, "-", color=color, linewidth=2.6, zorder=5)
    ax.plot(xs, ys, ".", color=color, markersize=4, zorder=6)


def _mark_st(ax, source, target, cols):
    for node, label in ((source, T("src_mark")), (target, T("tgt_mark"))):
        ax.text(node % cols, node // cols, label, ha="center", va="center",
                color="white", fontweight="bold", fontsize=11, zorder=7)


# ========================================================================== #
# 1) Izgara kesfi: Dijkstra vs Bellman-Ford                                   #
# ========================================================================== #
def make_grid_exploration(rows=26, cols=26, seed=GLOBAL_SEED):
    print(T("step1"))
    graph = gen.grid_2d(rows, cols, seed=seed, wmin=1, wmax=6)
    source = 3 * cols + 3
    target = (rows - 4) * cols + (cols - 4)

    dij = trace_dijkstra(graph, source)
    bf = trace_bellman_ford(graph, source, edge_order="reverse")
    path = reconstruct(dij["pred"], source, target)

    # ortak renk olcekleri
    cmap_dij = ListedColormap([C_UNSEEN, C_FRONTIER, C_DONE])
    cmap_bf = plt.cm.viridis.copy()
    cmap_bf.set_bad(C_UNSEEN)
    vmax = max(d for d in bf["snapshots"][-1] if d != INF)

    def render(ax_d, ax_b, frac, show_path):
        step = round(frac * dij["total"])
        p = round(frac * bf["passes"])
        ax_d.clear(); ax_b.clear()
        ax_d.imshow(dijkstra_state(dij, rows, cols, step),
                    cmap=cmap_dij, vmin=0, vmax=2, interpolation="nearest")
        ax_b.imshow(bf_field(bf, rows, cols, p), cmap=cmap_bf,
                    vmin=0, vmax=vmax, interpolation="nearest")
        ax_d.set_title(T("dij_title").format(step=step, total=dij["total"]),
                       fontsize=10)
        ax_b.set_title(T("bf_title").format(p=p, passes=bf["passes"],
                                            edges=bf["edges"]),
                       fontsize=10)
        for ax in (ax_d, ax_b):
            ax.set_xticks([]); ax.set_yticks([])
            _mark_st(ax, source, target, cols)
        if show_path:
            _draw_path(ax_d, path, cols)
            _draw_path(ax_b, path, cols)

    # ---- statik anlik goruntuler PNG'si (rapora/sunuma girer) ----
    fracs = [0.12, 0.32, 0.55, 0.78, 1.0]
    fig, axes = plt.subplots(2, len(fracs), figsize=(3.0 * len(fracs), 6.4))
    for col_i, fr in enumerate(fracs):
        render(axes[0, col_i], axes[1, col_i], fr, show_path=(fr >= 0.999))
        axes[0, col_i].set_title(T("pct").format(p=int(fr * 100)), fontsize=11)
        axes[1, col_i].set_title("")
    axes[0, 0].set_ylabel(T("ylabel_dij"), fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel(T("ylabel_bf"), fontsize=12, fontweight="bold")
    fig.suptitle(T("grid_suptitle"), fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    snap_path = os.path.join(_plots_dir(), "F_exploration_snapshots.png")
    fig.savefig(snap_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(T("written").format(path=snap_path))

    # ---- animasyonlu GIF ----
    n_frames = 46
    hold = 8
    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(12.4, 6.0))

    def update(i):
        idx = min(i, n_frames - 1)
        frac = idx / (n_frames - 1)
        render(ax_d, ax_b, frac, show_path=(i >= n_frames - 1))
        return []

    try:
        anim = FuncAnimation(fig, update, frames=n_frames + hold,
                             interval=140, blit=False)
        gif_path = os.path.join(_anim_dir(), "grid_exploration.gif")
        anim.save(gif_path, writer=PillowWriter(fps=7))
        print(T("written").format(path=gif_path))
    except Exception as exc:  # noqa: BLE001
        print(T("gif_fail").format(exc=repr(exc)))
    plt.close(fig)


# ========================================================================== #
# 2) Floyd-Warshall: uzaklik matrisinin dolmasi                               #
# ========================================================================== #
def make_floyd_warshall_matrix(n=64, seed=GLOBAL_SEED):
    print(T("step2"))
    graph = gen.random_sparse(n, avg_degree=4, seed=seed)
    tr = trace_floyd_warshall(graph)
    snaps, labels = tr["snapshots"], tr["labels"]

    cmap = plt.cm.plasma.copy()
    cmap.set_bad(C_UNSEEN)
    finite_final = snaps[-1][np.isfinite(snaps[-1])]
    vmax = float(np.percentile(finite_final, 99)) if finite_final.size else 1.0

    def masked(D):
        return np.where(np.isfinite(D), D, np.nan)

    def filled_fraction(D):
        return np.isfinite(D).sum() / D.size

    # ---- statik anlik goruntuler PNG'si ----
    pick = [0, len(snaps) // 4, len(snaps) // 2, 3 * len(snaps) // 4,
            len(snaps) - 1]
    fig, axes = plt.subplots(1, len(pick), figsize=(3.05 * len(pick), 3.5))
    for ax, i in zip(axes, pick):
        ax.imshow(masked(snaps[i]), cmap=cmap, vmin=0, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(T("fw_snap_title").format(
                         k=labels[i],
                         pct=f"{filled_fraction(snaps[i])*100:.0f}"),
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(T("fw_suptitle").format(n=n), fontsize=12,
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    snap_path = os.path.join(_plots_dir(), "F_floyd_warshall_snapshots.png")
    fig.savefig(snap_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(T("written").format(path=snap_path))

    # ---- animasyonlu GIF ----
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    im = ax.imshow(masked(snaps[0]), cmap=cmap, vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(T("fw_cbar"))

    def update(i):
        idx = min(i, len(snaps) - 1)
        im.set_data(masked(snaps[idx]))
        ax.set_title(T("fw_gif_title").format(
                         k=labels[idx], n=tr["n"],
                         pct=f"{filled_fraction(snaps[idx])*100:.0f}"),
                     fontsize=11)
        return [im]

    try:
        anim = FuncAnimation(fig, update, frames=len(snaps) + 8,
                             interval=130, blit=False)
        gif_path = os.path.join(_anim_dir(), "floyd_warshall_matrix.gif")
        anim.save(gif_path, writer=PillowWriter(fps=8))
        print(T("written").format(path=gif_path))
    except Exception as exc:  # noqa: BLE001
        print(T("gif_fail").format(exc=repr(exc)))
    plt.close(fig)


def main():
    global LANG
    LANG = _parse_lang(sys.argv[1:])
    ensure_dirs()
    print(T("gen"))
    make_grid_exploration()
    make_floyd_warshall_matrix()
    print(T("done"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

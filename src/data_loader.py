"""
data_loader.py -- Gercek dunya veri kumesinin yuklenmesi ve is;lenmesi.

Veri kumesi: 9. DIMACS Implementation Challenge -- Shortest Paths.
New York sehri yol agi (USA-road-d.NY): 264.346 dugum, 733.846 yonlu
yay (arc). Kenar agirliklari gercek yol mesafeleridir.

Kaynak: http://www.diag.uniroma1.it/challenge9/

DIMACS .gr dosya bicimi:
    c ... yorum satiri ...
    p sp <dugum_sayisi> <yay_sayisi>
    a <kaynak> <hedef> <agirlik>
Dugumler dosyada 1'den baslar; bu modul 0-tabanli indekse cevirir.

Bu modul ayrica gercek ag uzerinden KONTROLLU BOYUTTA bagli alt-graflar
ornekler; boylece dort algoritma da gercek yol agi yapisi uzerinde
karsilastirilabilir (tam graf, komsuluk matrisi tabanli yaklasimlar icin
fazla buyuktur).
"""
from __future__ import annotations

import os
import random
from collections import deque

from config import DIMACS_NY_GR, DIMACS_NY_URL, DATA_RAW_DIR
from graph import Graph


def ensure_dimacs_ny():
    """
    DIMACS NY yol agi dosyasinin yerel diskte oldugundan emin olur;
    yoksa internetten indirir ve acar. Proje boylece kendi kendine
    yeterlidir. Var olan dosya icin hicbir sey yapmaz.
    """
    if os.path.exists(DIMACS_NY_GR):
        return DIMACS_NY_GR

    import urllib.request
    import gzip
    import shutil

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    gz_path = DIMACS_NY_GR + ".gz"
    print(f"  DIMACS NY yol agi indiriliyor:\n    {DIMACS_NY_URL}")
    urllib.request.urlretrieve(DIMACS_NY_URL, gz_path)
    print("  Acaliyor (gunzip)...")
    with gzip.open(gz_path, "rb") as fin, open(DIMACS_NY_GR, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    print(f"  Hazir: {DIMACS_NY_GR}")
    return DIMACS_NY_GR


def load_dimacs(path=None, name="NY-yol-agi", max_arcs=None):
    """
    DIMACS .gr dosyasini okuyup bir Graph nesnesi dondurur.

    max_arcs verilirse yalnizca ilk o kadar yay okunur (hizli testler
    icin); normalde None birakilir.
    """
    if path is None:
        path = DIMACS_NY_GR
    if not os.path.exists(path):
        ensure_dimacs_ny()

    n = 0
    edges = []
    with open(path, "r") as f:
        for line in f:
            if not line:
                continue
            tag = line[0]
            if tag == "a":
                # "a <u> <v> <w>"
                _, u, v, w = line.split()
                edges.append((int(u) - 1, int(v) - 1, float(w)))
                if max_arcs is not None and len(edges) >= max_arcs:
                    break
            elif tag == "p":
                # "p sp <n> <m>"
                parts = line.split()
                n = int(parts[2])
            # "c" ile baslayan yorum satirlari atlanir

    # DIMACS dosyasi her iki yonu de ayri yay olarak verir -> directed=True
    return Graph(n, edges, directed=True, name=name)


def sample_connected_subgraph(graph, k, seed=0):
    """
    Buyuk bir graftan ~k dugumlu BAGLI bir alt-graf ornekler.

    Yontem: rastgele bir baslangic dugumunden genislik-oncelikli arama
    (BFS) ile k dugum toplanir; sonra bu dugumler uzerinde indirgenmis
    alt-graf (induced subgraph) cikarilir ve dugumler 0..k-1 olarak
    yeniden adlandirilir.

    Boylece dort algoritma da kucuk, kontrollu boyutta ama GERCEK yol agi
    yapisina sahip graflar uzerinde calistirilabilir.
    """
    rng = random.Random(seed)
    adj = graph.adjacency_list()
    n = graph.n
    k = min(k, n)

    # Komsusu olan bir baslangic dugumu sec
    start = rng.randrange(n)
    guard = 0
    while not adj[start] and guard < 1000:
        start = rng.randrange(n)
        guard += 1

    # BFS ile k dugum topla
    visited = {start}
    order = [start]
    queue = deque([start])
    while queue and len(order) < k:
        u = queue.popleft()
        neighbors = [v for (v, _) in adj[u]]
        rng.shuffle(neighbors)
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                order.append(v)
                queue.append(v)
                if len(order) >= k:
                    break

    # Indirgenmis alt-graf: yalnizca secilen dugumler arasindaki kenarlar
    sub_nodes = order
    relabel = {old: new for new, old in enumerate(sub_nodes)}
    sub_set = visited
    edges = []
    for u in sub_nodes:
        ru = relabel[u]
        for (v, w) in adj[u]:
            if v in sub_set:
                edges.append((ru, relabel[v], w))

    return Graph(len(sub_nodes), edges, directed=True,
                 name=f"NY-altgraf(n={len(sub_nodes)})")


if __name__ == "__main__":
    # Hizli kendi kendine test
    ensure_dimacs_ny()
    g = load_dimacs()
    print("Yuklendi:", g)
    sub = sample_connected_subgraph(g, 1000, seed=1)
    print("Alt-graf:", sub, "-> describe:", sub.describe())

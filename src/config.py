"""
config.py -- Proje genelinde kullanilan yol sabitleri ve deney ayarlari.

Bu dosya tek bir yerden tum dizin yollarini ve deney parametrelerini
yonetir. Deney boyutlari iki profilde tanimlidir:

  * "quick" : hizli demo / gelistirme icin -- saniyeler surer.
  * "full"  : rapora giren tam sonuclar icin -- birkac dakika surebilir.

Boyut listeleri ozellikle, her algoritmanin "pratik siniri" gorunecek
sekilde secilmistir (ornegin Floyd-Warshall O(V^3) oldugu icin daha
kucuk dugum sayilarinda durur). Bir algoritmanin daha buyuk girdilerde
"infeasible" (uygulanamaz) hale gelmesi de bir deney sonucudur.
"""
import os

# --------------------------------------------------------------------------
# Dizin yollari (hepsi proje kokune gore otomatik hesaplanir)
# --------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_GEN_DIR = os.path.join(DATA_DIR, "generated")

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
RESULTS_RAW_DIR = os.path.join(RESULTS_DIR, "raw")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
ANIM_DIR = os.path.join(RESULTS_DIR, "animations")

REPORT_DIR = os.path.join(ROOT_DIR, "report")
SLIDES_DIR = os.path.join(ROOT_DIR, "presentation")

# Ingilizce surum ciktilari -- ayri klasorler, Turkce ciktilarin uzerine
# yazmaz. plots.py / animate.py "--lang en" ile bu klasorlere yazar.
PLOTS_DIR_EN = os.path.join(RESULTS_DIR, "plots_en")
TABLES_DIR_EN = os.path.join(RESULTS_DIR, "tables_en")
ANIM_DIR_EN = os.path.join(RESULTS_DIR, "animations_en")

# Gercek veri kumesi (DIMACS 9. Implementation Challenge -- New York yol agi)
DIMACS_NY_GR = os.path.join(DATA_RAW_DIR, "USA-road-d.NY.gr")
DIMACS_NY_URL = (
    "http://www.diag.uniroma1.it/challenge9/data/"
    "USA-road-d/USA-road-d.NY.gr.gz"
)

# Benchmark sonuclarinin yazildigi tek JSON dosyasi
RESULTS_JSON = os.path.join(RESULTS_RAW_DIR, "benchmark_results.json")

# --------------------------------------------------------------------------
# Sabitler
# --------------------------------------------------------------------------
INF = float("inf")

# Sentetik graflarda kenar agirlik araligi (yol mesafelerini taklit eder)
WEIGHT_MIN = 1
WEIGHT_MAX = 1000

# Tekrarlanabilirlik icin ana tohum (seed)
GLOBAL_SEED = 20252  # 2025 bahar donemi

# Tek bir benchmark kosusu icin guvenlik butcesi (saniye). Bir konfigurasyon
# bunu asacaksa benchmark o algoritmayi daha buyuk girdilerde calistirmaz.
TIME_BUDGET_SEC = 90.0

# --------------------------------------------------------------------------
# Deney profilleri
# --------------------------------------------------------------------------
# Her algoritmanin benchmark'ta calistirilacagi en buyuk dugum sayisi.
# Bu sinirlar teorik karmasiklik + bellek kisitlarindan turetilmistir:
#   - dijkstra_array ve floyd_warshall O(V^2) bellek (komsuluk matrisi) ister.
#   - floyd_warshall ayrica O(V^3) zaman ister -> en dusuk sinir.
#   - bellman_ford O(V*E); buyuk graflarda pratik degildir.
ALGO_NODE_CAP = {
    "quick": {
        "dijkstra_heap": 1000,
        "dijkstra_array": 1000,
        "bellman_ford": 1000,
        "floyd_warshall": 400,
    },
    "full": {
        "dijkstra_heap": 12800,
        "dijkstra_array": 12800,
        "bellman_ford": 6400,
        "floyd_warshall": 2500,
    },
}

EXPERIMENT_CONFIG = {
    "quick": {
        # Deney A -- seyrek graflarda dugum sayisina gore olceklenme
        "scaling_sizes": [64, 128, 256, 512, 1000],
        "scaling_avg_degree": 4,
        # Deney B -- yogunlugun etkisi (dugum sayisi sabit)
        "density_n": 256,
        "density_fracs": [0.03, 0.08, 0.15, 0.30],
        # Deney C -- on-isleme / sorgu sayisi odunlesimi
        "manyq_n": 256,
        "manyq_avg_degree": 4,
        "manyq_counts": [1, 5, 20, 100, 500, 2000],
        # Deney D -- gercek yol agi
        "real_subgraph_sizes": [128, 256, 512],
        "real_full_queries": 5,
        # Olcum tekrar sayilari
        "sources": 3,   # her graf icin kac farkli kaynak dugum
        "repeats": 2,   # her olcum kac kez tekrarlanir (medyan alinir)
    },
    "full": {
        "scaling_sizes": [100, 200, 400, 800, 1600, 3200, 6400, 12800],
        "scaling_avg_degree": 4,
        "density_n": 1000,
        "density_fracs": [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50],
        "manyq_n": 1000,
        "manyq_avg_degree": 4,
        "manyq_counts": [1, 5, 20, 100, 500, 2000, 10000, 50000],
        "real_subgraph_sizes": [500, 1000, 2000, 4000],
        "real_full_queries": 20,
        "sources": 5,
        "repeats": 3,
    },
}

# Algoritmalarin rapor/grafiklerde kullanilan okunabilir adlari ve renkleri
ALGO_LABELS = {
    "dijkstra_heap": "Dijkstra (ikili yigin + komsuluk listesi)",
    "dijkstra_array": "Dijkstra (dizi taramasi + komsuluk matrisi)",
    "bellman_ford": "Bellman-Ford (kenar listesi)",
    "floyd_warshall": "Floyd-Warshall (komsuluk matrisi)",
}

ALGO_SHORT = {
    "dijkstra_heap": "Dijkstra-Yigin",
    "dijkstra_array": "Dijkstra-Dizi",
    "bellman_ford": "Bellman-Ford",
    "floyd_warshall": "Floyd-Warshall",
}

# Ingilizce kisa adlar -- rapor/grafiklerin Ingilizce surumu icin
ALGO_SHORT_EN = {
    "dijkstra_heap": "Dijkstra-Heap",
    "dijkstra_array": "Dijkstra-Array",
    "bellman_ford": "Bellman-Ford",
    "floyd_warshall": "Floyd-Warshall",
}

ALGO_COLORS = {
    "dijkstra_heap": "#1f77b4",
    "dijkstra_array": "#ff7f0e",
    "bellman_ford": "#2ca02c",
    "floyd_warshall": "#d62728",
}


def ensure_dirs():
    """Cikti dizinlerinin var oldugundan emin olur."""
    for d in (DATA_RAW_DIR, DATA_GEN_DIR, RESULTS_RAW_DIR,
              PLOTS_DIR, TABLES_DIR, ANIM_DIR, REPORT_DIR, SLIDES_DIR,
              PLOTS_DIR_EN, TABLES_DIR_EN, ANIM_DIR_EN):
        os.makedirs(d, exist_ok=True)

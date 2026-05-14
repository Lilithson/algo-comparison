"""
algorithms paketi -- dort en-kisa-yol yaklasiminin gerceklenmesi.

Her yaklasim ayri bir modulde tutulur (raporun istedigi "farkli
yaklasimlar icin ayri gerceklemeler" ilkesi):

  dijkstra_heap   -- Dijkstra + ikili yigin + komsuluk listesi
  dijkstra_array  -- Dijkstra + lineer dizi taramasi + komsuluk matrisi
  bellman_ford    -- Bellman-Ford + kenar listesi
  floyd_warshall  -- Floyd-Warshall (tum ciftler) + komsuluk matrisi
"""
from algorithms.common import (
    SSSPResult, APSPResult, reconstruct_path, reconstruct_path_fw,
    query_distance,
)
from algorithms.dijkstra_heap import dijkstra_heap
from algorithms.dijkstra_array import dijkstra_array
from algorithms.bellman_ford import bellman_ford
from algorithms.floyd_warshall import floyd_warshall

# Tek-kaynakli algoritmalarin ad -> fonksiyon eslemesi
SSSP_ALGORITHMS = {
    "dijkstra_heap": dijkstra_heap,
    "dijkstra_array": dijkstra_array,
    "bellman_ford": bellman_ford,
}

__all__ = [
    "SSSPResult", "APSPResult", "reconstruct_path", "reconstruct_path_fw",
    "query_distance", "dijkstra_heap", "dijkstra_array", "bellman_ford",
    "floyd_warshall", "SSSP_ALGORITHMS",
]

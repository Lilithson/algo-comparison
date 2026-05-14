# Ağırlıklı Graflarda En Kısa Yol Sorguları
### Algoritma II — Dönem Sonu Projesi · Karşılaştırmalı Algoritma Tasarımı ve Karmaşıklık Analizi

Bu proje, **aynı problemi** — ağırlıklı bir graf üzerinde en kısa yol
sorgularını verimli yanıtlamayı — **dört farklı algoritmik yaklaşımla**
çözer, her birini teorik olarak analiz eder ve hem sentetik hem de
gerçek veri (DIMACS New York yol ağı) üzerinde deneysel olarak
karşılaştırır.

Amaç yalnızca doğru cevabı bulmak değil; graf gösterimi, algoritmik
strateji ve tasarım felsefesi seçimlerinin **performans, ölçeklenme ve
bellek** üzerindeki etkisini anlamaktır.

---

## Hızlı Başlangıç

```bash
pip install -r requirements.txt      # numpy, matplotlib, python-docx, python-pptx

bash run_all.sh                      # tam boru hattı (rapor sonuçları; birkaç dakika)
#   ya da:
bash run_all.sh quick                # hızlı profil (demo / geliştirme; ~yarım dakika)
```

`run_all.sh` sırasıyla şunları yapar: doğruluk testleri → benchmark
deneyleri → grafikler ve tablolar → Word raporu → PowerPoint sunumu →
PDF dönüşümü. Grafikler, rapor ve sunum hem **Türkçe** hem **İngilizce**
üretilir.

> Gerçek veri kümesi (DIMACS New York yol ağı) ilk çalıştırmada
> `data_loader.py` tarafından **otomatik indirilir** (~3,7 MB).

---

## Dört Yaklaşım

| # | Yaklaşım | Graf gösterimi | Zaman | Bellek | Tasarım felsefesi |
|---|----------|----------------|-------|--------|-------------------|
| 1 | Dijkstra + İkili Yığın | Komşuluk listesi | O((V+E) log V) | O(V+E) | Talep-üzerine, zaman-optimize |
| 2 | Dijkstra + Dizi Tarama | Komşuluk matrisi | O(V²) | O(V²) | Yoğun graf-optimize, basit |
| 3 | Bellman-Ford | Kenar listesi | O(V·E) | O(V+E) | Genellik (negatif ağırlık) |
| 4 | Floyd-Warshall | Komşuluk matrisi | O(V³) ön-işleme, O(1) sorgu | O(V²) | Ağır ön-işleme, hızlı sorgu |

Üç graf gösteriminin tamamı ve dört farklı strateji kapsanır. Dört
kişilik gruba doğal dağılım: her üyeye bir yaklaşım.

---

## Deneyler

| Deney | Sorusu |
|-------|--------|
| **A — Ölçeklenme** | Düğüm sayısı V arttıkça süre nasıl değişir? |
| **B — Yoğunluk** | V sabitken kenar yoğunluğu süreyi nasıl etkiler? |
| **C — Sorgu sayısı** | Ön-işleme maliyeti kaç sorguda amorti olur? (başabaş noktası) |
| **D — Gerçek yol ağı** | 264 bin düğümlü gerçek graf üzerinde hangi yaklaşımlar uygulanabilir? |

Her ölçümde hem çalışma süresi (`time.perf_counter`, çoklu kaynak +
tekrar, medyan) hem de tepe bellek (`tracemalloc`) kaydedilir. Ayrıca
donanımdan bağımsız bir doğrulama için yapılan iş miktarı (gevşetme
sayısı) sayılır.

---

## Dizin Yapısı

```
algo/
├── run_all.sh                 Tek komutluk tam boru hattı
├── requirements.txt           Python bağımlılıkları
├── README.md
│
├── src/
│   ├── config.py              Yol sabitleri ve deney profilleri
│   ├── graph.py               Graf sınıfı + üç gösterim biçimi
│   ├── generators.py          Sentetik graf üreteçleri (seyrek/yoğun/ızgara/ölçeksiz/DAG)
│   ├── data_loader.py         DIMACS gerçek veri okuyucu + alt-graf örnekleyici
│   ├── algorithms/
│   │   ├── common.py          Ortak sonuç tipleri + yol geri kurma
│   │   ├── dijkstra_heap.py   Yaklaşım 1
│   │   ├── dijkstra_array.py  Yaklaşım 2
│   │   ├── bellman_ford.py    Yaklaşım 3
│   │   └── floyd_warshall.py  Yaklaşım 4 (+ saf Python referans sürümü)
│   ├── verify.py              Doğruluk test paketi
│   ├── benchmark.py           Deney koşucusu
│   ├── plots.py               Grafik ve tablo üretici (--lang en: İngilizce)
│   ├── animate.py             Görsel simülasyon üretici (--lang en: İngilizce)
│   ├── demo.py                Canlı karşılaştırma demosu (komut satırı)
│
├── data/raw/                  İndirilen DIMACS veri kümesi
├── results/
│   ├── raw/                   Ham benchmark sonuçları (JSON)
│   ├── plots/  plots_en/      Karşılaştırma grafikleri + simülasyon kareleri (PNG)
│   ├── tables/  tables_en/    4 sayısal tablo (CSV + Markdown)
│   └── animations/  animations_en/   Görsel simülasyonlar (GIF)
├── report/                    Report.docx/pdf (İngilizce)
└── presentation/              Presentation.pptx/pdf (İngilizce)
```

---

## Adım Adım Çalıştırma

`run_all.sh` yerine adımlar tek tek de çalıştırılabilir:

```bash
python src/verify.py             # 1) Doğruluk: 4 algoritma + referans çapraz kontrol
python src/benchmark.py --full   # 2) Deneyler -> results/raw/benchmark_results.json
python src/plots.py              # 3) Grafikler + tablolar -> results/
python src/animate.py            # 4) Görsel simülasyonlar -> results/animations/ + plots/

### Canlı demo ve görsel simülasyon

İki ek araç, farkları "çalışırken" göstermek içindir:

```bash
# Canlı karşılaştırma — dört algoritmayı aynı (s,t) sorgusunda yan yana çalıştırır;
# bulunan yolu, süreyi, yapılan iş miktarını ve belleği tek tabloda gösterir:
python src/demo.py                              # varsayılan: 20x20 ızgara
python src/demo.py --kind sparse --n 3000        # büyük seyrek graf
python src/demo.py --kind grid --n 25 --trace    # küçük graf + adım adım iz

# Görsel simülasyon — algoritmaları grafı keşfederken izleten animasyonlar:
python src/animate.py
#   results/animations/grid_exploration.gif      Dijkstra vs Bellman-Ford (yan yana)
#   results/animations/floyd_warshall_matrix.gif Floyd-Warshall matrisinin dolması
#   results/plots/F_*_snapshots.png              rapora/sunuma giren statik kareler
```

`benchmark.py` iki profil destekler:

* `--quick` : küçük girdiler, hızlı; geliştirme ve demo için.
* `--full`  : rapora giren tam sonuçlar; birkaç dakika sürer.

Tüm sonuçlar sabit tohum (seed) ile **yeniden üretilebilir**
(`config.GLOBAL_SEED`).

---

## Gereksinimler

* **Python 3.10+** ve `numpy`, `matplotlib` (algoritmalar yalnızca
  standart kütüphane + numpy kullanır; matplotlib grafikler içindir).

---

## Veri Kümesi

**9. DIMACS Implementation Challenge — Shortest Paths**, New York şehri
yol ağı (`USA-road-d.NY`): 264.346 düğüm, 733.846 yönlü yay; kenar
ağırlıkları gerçek yol mesafeleridir.
Kaynak: <http://www.diag.uniroma1.it/challenge9/>

Sentetik graflar ise deney boyutunu tam kontrol etmek için
`generators.py` tarafından üretilir.

---

## Teslim Edilenler

1. **Kaynak kod** — `src/` (dört yaklaşım ayrı modüllerde, belgelenmiş).
2. **Yazılı rapor** — `report/Report.docx` ve `report/Report.pdf`
3. **Sunum** — `presentation/Presentation.pptx` ve `presentation/Presentation.pdf`
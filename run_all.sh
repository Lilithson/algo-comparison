#!/usr/bin/env bash
# ===========================================================================
# run_all.sh  --  Tum projeyi tek komutla calistirir.
#
#   bash run_all.sh          # tam profil  (rapora giren sonuclar; birkac dakika)
#   bash run_all.sh quick    # hizli profil (demo / gelistirme; ~yarim dakika)
#
# Sirasiyla: dogrulama -> benchmark -> grafik/tablo -> gorsel simulasyon
# ===========================================================================
set -e
cd "$(dirname "$0")"

PROFILE="${1:-full}"
if [ "$PROFILE" = "quick" ]; then
    BENCH_FLAG="--quick"
else
    PROFILE="full"
    BENCH_FLAG="--full"
fi

echo "############################################################"
echo "#  Agirlikli Graflarda En Kisa Yol -- tam boru hatti"
echo "#  Profil: $PROFILE"
echo "############################################################"

echo
echo ">>> [1/4] Dogruluk testleri (verify.py)..."
python3 src/verify.py

echo
echo ">>> [2/4] Benchmark deneyleri (benchmark.py $BENCH_FLAG)..."
python3 src/benchmark.py "$BENCH_FLAG"

echo
echo ">>> [3/4] Grafikler ve tablolar (plots.py) -- TR + EN..."
python3 src/plots.py
python3 src/plots.py --lang en

echo
echo ">>> [4/4] Gorsel simulasyonlar (animate.py) -- TR + EN..."
python3 src/animate.py
python3 src/animate.py --lang en

echo
echo "############################################################"
echo "#  TAMAMLANDI. Ciktilar:"
echo "#    results/plots/      -- 9 grafik (.png) -- karsilastirma + simulasyon"
echo "#    results/tables/     -- 4 sayisal tablo (.csv + .md)"
echo "#    results/animations/ -- 2 animasyon (.gif)"
echo "#    results/*_en/       -- yukaridakilerin Ingilizce surumu (plots_en, ...)"
echo "#    results/raw/        -- ham benchmark sonuclari (.json)"
echo "#"
echo "#  Canli demo:  python src/demo.py"
echo "############################################################"

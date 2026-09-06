#!/usr/bin/env bash
# analyze_task2b.sh
#
# Produces the CSVs needed for the Task 2 report (README.md sec 5):
#   1. speedup vs matrix size            (simd / prefetch / optimized vs naive)
#   2. speedup vs prefetch distance      (matmul_prefetch only)
#   3. speedup vs prefetch hint level    (matmul_prefetch only)
#   4. speedup vs SIMD width             (separate ungraded SSE/AVX2/AVX512 files)
#
# Parts 2-4 need small additions to your own source before they'll produce
# real numbers -- see the comments in each section. Part 1 works today as-is.
#
# Usage: ./analyze_task2b.sh [all|size|prefetch-dist|prefetch-hint|simd-width]

set -euo pipefail
cd "$(dirname "$0")"

OUT=results
mkdir -p "$OUT"

WHAT="${1:-all}"

# ---------------------------------------------------------------------------
# Helper: run `./bin/matmul all M M K seed` and append one CSV row per
# scored stage (simd/prefetch/optimized -- single-word names, easy to grep).
# ---------------------------------------------------------------------------
run_size() {
    local n="$1" seed="${2:-1234}"
    ./bin/matmul all "$n" "$n" "$n" "$seed" \
        | awk -v n="$n" '
            /^simd|^prefetch|^optimized/ {
                gsub(/x$/, "", $5)          # strip trailing "x" off speedup
                print n","$1","$3","$4","$5
            }'
}

# ---------------------------------------------------------------------------
# 1) Speedup vs matrix size
# ---------------------------------------------------------------------------
size_sweep() {
    echo "== 1/4: speedup vs matrix size =="
    make -s
    local csv="$OUT/size_sweep.csv"
    echo "size,stage,time_ms,gflops,speedup" > "$csv"
    for n in 128 256 384 512 768 1024 1536 2048; do
        echo "  N=$n"
        run_size "$n" >> "$csv"
    done
    echo "  -> $csv"
}

# ---------------------------------------------------------------------------
# 2) Speedup vs prefetch distance
#
# Requires: in matmul_prefetch.cpp, gate the prefetch distance behind a macro
# with a default, e.g.:
#     #ifndef PF_DIST
#     #define PF_DIST 8            // floats ahead to prefetch
#     #endif
#     _mm_prefetch((const char*)(ptr + PF_DIST), _MM_HINT_T0);
# so it can be overridden from the command line with -DPF_DIST=<n>.
# ---------------------------------------------------------------------------
prefetch_dist_sweep() {
    echo "== 2/4: speedup vs prefetch distance =="
    local csv="$OUT/prefetch_distance_sweep.csv"
    echo "distance,time_ms,gflops,speedup" > "$csv"
    local n=2048  # pick a size well past cache capacity, per README
    for dist in 0 8 16 32 64 128 256; do
        echo "  PF_DIST=$dist"
        make -s clean
        make -s CXXFLAGS_EXTRA="-DPF_DIST=${dist}" \
             CXXFLAGS="-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall -DPF_DIST=${dist}"
        ./bin/matmul prefetch "$n" "$n" "$n" \
            | awk -v d="$dist" '
                /^prefetch/ {
                    gsub(/x$/, "", $5)
                    print d","$3","$4","$5
                }' >> "$csv"
    done
    make -s clean && make -s
    echo "  -> $csv"
}

# ---------------------------------------------------------------------------
# 3) Speedup vs prefetch hint level (_MM_HINT_T0/T1/T2/NTA)
#
# Requires the same PF_DIST-style trick but for the hint macro, e.g.:
#     #ifndef PF_HINT
#     #define PF_HINT _MM_HINT_T0
#     #endif
#     _mm_prefetch((const char*)(ptr + PF_DIST), PF_HINT);
# ---------------------------------------------------------------------------
prefetch_hint_sweep() {
    echo "== 3/4: speedup vs prefetch hint level =="
    local csv="$OUT/prefetch_hint_sweep.csv"
    echo "hint,time_ms,gflops,speedup" > "$csv"
    local n=2048
    for hint in _MM_HINT_T0 _MM_HINT_T1 _MM_HINT_T2 _MM_HINT_NTA; do
        echo "  PF_HINT=$hint"
        make -s clean
        make -s CXXFLAGS="-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall -DPF_HINT=${hint}"
        ./bin/matmul prefetch "$n" "$n" "$n" \
            | awk -v h="$hint" '
                /^prefetch/ {
                    gsub(/x$/, "", $5)
                    print h","$3","$4","$5
                }' >> "$csv"
    done
    make -s clean && make -s
    echo "  -> $csv"
}

# ---------------------------------------------------------------------------
# 4) Speedup vs SIMD width (SSE / AVX2 / AVX-512)
#
# README sec 5.4: these must live in SEPARATE ungraded files (do not touch
# the graded src/matmul_simd.cpp, which must stay AVX2-only). Create e.g.:
#     src/width_variants/matmul_simd_sse.cpp     (-msse4.2)
#     src/width_variants/matmul_simd_avx2.cpp    (copy of your real simd kernel)
#     src/width_variants/matmul_simd_avx512.cpp  (-mavx512f)
# each exporting a distinctly-named function (matmul_simd_sse / _avx2 / _avx512)
# declared in a small local header. This section compiles+links a throwaway
# binary per width against matmul_naive.cpp + main-less timing stub.
# Adjust SRC_DIR / function names once those files exist.
# ---------------------------------------------------------------------------
simd_width_sweep() {
    echo "== 4/4: speedup vs SIMD width =="
    local csv="$OUT/simd_width_sweep.csv"
    echo "width,flag,note" > "$csv"
    echo "  (skipped: create src/width_variants/matmul_simd_{sse,avx2,avx512}.cpp first)"
    echo "sse,-msse4.2,not-yet-implemented"   >> "$csv"
    echo "avx2,-mavx2 -mfma,not-yet-implemented" >> "$csv"
    echo "avx512,-mavx512f,not-yet-implemented"  >> "$csv"
    echo "  -> $csv (placeholder until width_variants/ exists)"
}

case "$WHAT" in
    size)           size_sweep ;;
    prefetch-dist)  prefetch_dist_sweep ;;
    prefetch-hint)  prefetch_hint_sweep ;;
    simd-width)     simd_width_sweep ;;
    all)
        size_sweep
        prefetch_dist_sweep
        prefetch_hint_sweep
        simd_width_sweep
        ;;
    *)
        echo "usage: $0 [all|size|prefetch-dist|prefetch-hint|simd-width]" >&2
        exit 1
        ;;
esac

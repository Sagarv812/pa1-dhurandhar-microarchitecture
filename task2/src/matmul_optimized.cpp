// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"

#define PREFETCH_HINT _MM_HINT_T0

#define KSTEP(p)                              \
    {                                         \
        __m256 vb0 = _mm256_loadu_ps(b0 + p); \
        __m256 vb1 = _mm256_loadu_ps(b1 + p); \
        __m256 vb2 = _mm256_loadu_ps(b2 + p); \
        __m256 va;                            \
        va = _mm256_loadu_ps(a0 + p);         \
        c00 = _mm256_fmadd_ps(va, vb0, c00);  \
        c01 = _mm256_fmadd_ps(va, vb1, c01);  \
        c02 = _mm256_fmadd_ps(va, vb2, c02);  \
        va = _mm256_loadu_ps(a1 + p);         \
        c10 = _mm256_fmadd_ps(va, vb0, c10);  \
        c11 = _mm256_fmadd_ps(va, vb1, c11);  \
        c12 = _mm256_fmadd_ps(va, vb2, c12);  \
        va = _mm256_loadu_ps(a2 + p);         \
        c20 = _mm256_fmadd_ps(va, vb0, c20);  \
        c21 = _mm256_fmadd_ps(va, vb1, c21);  \
        c22 = _mm256_fmadd_ps(va, vb2, c22);  \
        va = _mm256_loadu_ps(a3 + p);         \
        c30 = _mm256_fmadd_ps(va, vb0, c30);  \
        c31 = _mm256_fmadd_ps(va, vb1, c31);  \
        c32 = _mm256_fmadd_ps(va, vb2, c32);  \
    }

static float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int T = 96;
    const int prefetch_dist = 64;
    
    for (int jo = 0; jo < N; jo += T) {
        const int j_end = (jo + T < N)? jo + T : N;
        
        int i = 0;
        for(; i + 4 <= M; i+=4) {
            const float* a0 = A + static_cast<long>(i) * lda;
            const float* a1 = a0 + lda;
            const float* a2 = a1 + lda;
            const float* a3 = a2 + lda;

            int j = jo;
            for(; j + 3 <= j_end; j += 3) {
                const float* b0 = B + static_cast<long>(j) * ldb;
                const float* b1 = b0 + ldb;
                const float* b2 = b1 + ldb;

                __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
                __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
                __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
                __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c32 = _mm256_setzero_ps();
                
                int p = 0;
                for(; p + 16 <= K; p += 16) {
                    _mm_prefetch(b0 + p + prefetch_dist, PREFETCH_HINT);
                    _mm_prefetch(b1 + p + prefetch_dist, PREFETCH_HINT);
                    _mm_prefetch(b2 + p + prefetch_dist, PREFETCH_HINT);

                    KSTEP(p);
                    KSTEP(p+8);
                }
                for(; p + 8 <= K; p += 8) KSTEP(p);

                float s[4][3] = {{hsum256(c00), hsum256(c01), hsum256(c02)},
                                 {hsum256(c10), hsum256(c11), hsum256(c12)},
                                 {hsum256(c20), hsum256(c21), hsum256(c22)},
                                 {hsum256(c30), hsum256(c31), hsum256(c32)}};

                for(; p < K;  p++) {
                    const float* ap[4] = {a0, a1, a2, a3};
                    const float* bp[3] = {b0, b1, b2};
                    for (int u = 0; u < 4; u++) {
                        for (int v = 0; v < 3; v++) {
                            s[u][v] += ap[u][p] * bp[v][p];
                        }
                    }
                }


                for (int u = 0; u < 4; ++u) { 
                    float* c = C + static_cast<long>(i + u) * ldc + j;
                    c[0] = s[u][0];
                    c[1] = s[u][1];
                    c[2] = s[u][2];
                }
            }

            for(; j < j_end; j++) {
                const float* b = B + static_cast<long>(j) * ldb;
                __m256 d0 = _mm256_setzero_ps(), d1 = _mm256_setzero_ps();
                __m256 d2 = _mm256_setzero_ps(), d3 = _mm256_setzero_ps();
                int p = 0;
                for(; p + 8 <= K; p += 8) {
                    __m256 acc = _mm256_loadu_ps(b + p);
                    d0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), acc, d0);
                    d1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + p), acc, d1);
                    d2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + p), acc, d2);
                    d3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + p), acc, d3);
                }

                float t0 = hsum256(d0), t1 = hsum256(d1);
                float t2 = hsum256(d2), t3 = hsum256(d3);

                for(; p < K; p++) {
                    t0 += a0[p] * b[p];
                    t1 += a1[p] * b[p];
                    t2 += a2[p] * b[p];
                    t3 += a3[p] * b[p];
                }
                C[static_cast<long>(i + 0) * ldc + j] = t0;
                C[static_cast<long>(i + 1) * ldc + j] = t1;
                C[static_cast<long>(i + 2) * ldc + j] = t2;
                C[static_cast<long>(i + 3) * ldc + j] = t3;
                
            }
        }

        for (; i < M; i++) {
            const float* a = A + static_cast<long>(i) * lda;
            for (int j = jo; j < j_end; j++) {
                const float* b = B + static_cast<long>(j) * ldb;
                __m256 acc = _mm256_setzero_ps();
                int p = 0;
                for (; p + 8 <= K; p += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
                float s = hsum256(acc);
                for (; p < K; ++p) s += a[p] * b[p];
                C[static_cast<long>(i) * ldc + j] = s;
            }
        }
    }
}
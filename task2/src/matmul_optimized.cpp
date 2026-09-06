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

static inline __attribute__((always_inline)) float hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline __attribute__((always_inline)) __m256i tail_mask(int rem)
{
    __m256i lane_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i vrem = _mm256_set1_epi32(rem);
    return _mm256_cmpgt_epi32(vrem, lane_idx);
}

static inline __attribute__((always_inline)) float dot_masked(const float *a, const float *b, int length, int prefetch_dist)
{
    int p = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; p + 8 <= length; p += 8)
    {
        if (p + prefetch_dist < length)
        {
            _mm_prefetch(reinterpret_cast<const char *>(a + p + prefetch_dist), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(b + p + prefetch_dist), _MM_HINT_T0);
        }
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
    }
    int rem = length - p;
    if (rem > 0)
    {
        __m256i mask = tail_mask(rem);
        __m256 av = _mm256_maskload_ps(a + p, mask);
        __m256 bv = _mm256_maskload_ps(b + p, mask);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    return hsum256(acc);
}

static inline void kernel_3x4(const float *a0, const float *a1, const float *a2,
                               const float *b0, const float *b1, const float *b2, const float *b3,
                               int K, int block_k, int prefetch_dist,
                               float *c0, float *c1, float *c2)
{
    __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps(), acc02 = _mm256_setzero_ps(), acc03 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps(), acc12 = _mm256_setzero_ps(), acc13 = _mm256_setzero_ps();
    __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps(), acc22 = _mm256_setzero_ps(), acc23 = _mm256_setzero_ps();

    for (int tk = 0; tk < K; tk += block_k)
    {
        const int tile_k = (tk + block_k < K) ? block_k : K - tk;
        const float *ca0 = a0 + tk, *ca1 = a1 + tk, *ca2 = a2 + tk;
        const float *cb0 = b0 + tk, *cb1 = b1 + tk, *cb2 = b2 + tk, *cb3 = b3 + tk;

        int p = 0;
        for (; p + 8 <= tile_k; p += 8)
        {
            if (p + prefetch_dist < tile_k)
            {
                _mm_prefetch(reinterpret_cast<const char *>(ca0 + p + prefetch_dist), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(ca1 + p + prefetch_dist), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(ca2 + p + prefetch_dist), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(cb0 + p + prefetch_dist), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(cb1 + p + prefetch_dist), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(cb2 + p + prefetch_dist), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(cb3 + p + prefetch_dist), _MM_HINT_T0);
            }
            __m256 av0 = _mm256_loadu_ps(ca0 + p);
            __m256 av1 = _mm256_loadu_ps(ca1 + p);
            __m256 av2 = _mm256_loadu_ps(ca2 + p);

            __m256 bv = _mm256_loadu_ps(cb0 + p);
            acc00 = _mm256_fmadd_ps(av0, bv, acc00);
            acc10 = _mm256_fmadd_ps(av1, bv, acc10);
            acc20 = _mm256_fmadd_ps(av2, bv, acc20);

            bv = _mm256_loadu_ps(cb1 + p);
            acc01 = _mm256_fmadd_ps(av0, bv, acc01);
            acc11 = _mm256_fmadd_ps(av1, bv, acc11);
            acc21 = _mm256_fmadd_ps(av2, bv, acc21);

            bv = _mm256_loadu_ps(cb2 + p);
            acc02 = _mm256_fmadd_ps(av0, bv, acc02);
            acc12 = _mm256_fmadd_ps(av1, bv, acc12);
            acc22 = _mm256_fmadd_ps(av2, bv, acc22);

            bv = _mm256_loadu_ps(cb3 + p);
            acc03 = _mm256_fmadd_ps(av0, bv, acc03);
            acc13 = _mm256_fmadd_ps(av1, bv, acc13);
            acc23 = _mm256_fmadd_ps(av2, bv, acc23);
        }

        int rem = tile_k - p;
        if (rem > 0)
        {
            __m256i mask = tail_mask(rem);
            __m256 av0 = _mm256_maskload_ps(ca0 + p, mask);
            __m256 av1 = _mm256_maskload_ps(ca1 + p, mask);
            __m256 av2 = _mm256_maskload_ps(ca2 + p, mask);

            __m256 bv = _mm256_maskload_ps(cb0 + p, mask);
            acc00 = _mm256_fmadd_ps(av0, bv, acc00);
            acc10 = _mm256_fmadd_ps(av1, bv, acc10);
            acc20 = _mm256_fmadd_ps(av2, bv, acc20);

            bv = _mm256_maskload_ps(cb1 + p, mask);
            acc01 = _mm256_fmadd_ps(av0, bv, acc01);
            acc11 = _mm256_fmadd_ps(av1, bv, acc11);
            acc21 = _mm256_fmadd_ps(av2, bv, acc21);

            bv = _mm256_maskload_ps(cb2 + p, mask);
            acc02 = _mm256_fmadd_ps(av0, bv, acc02);
            acc12 = _mm256_fmadd_ps(av1, bv, acc12);
            acc22 = _mm256_fmadd_ps(av2, bv, acc22);

            bv = _mm256_maskload_ps(cb3 + p, mask);
            acc03 = _mm256_fmadd_ps(av0, bv, acc03);
            acc13 = _mm256_fmadd_ps(av1, bv, acc13);
            acc23 = _mm256_fmadd_ps(av2, bv, acc23);
        }
    }

    c0[0] = hsum256(acc00); c0[1] = hsum256(acc01); c0[2] = hsum256(acc02); c0[3] = hsum256(acc03);
    c1[0] = hsum256(acc10); c1[1] = hsum256(acc11); c1[2] = hsum256(acc12); c1[3] = hsum256(acc13);
    c2[0] = hsum256(acc20); c2[1] = hsum256(acc21); c2[2] = hsum256(acc22); c2[3] = hsum256(acc23);
}

void matmul_optimized(const float *A, const float *B, float *C,
                       int M, int N, int K, int lda, int ldb, int ldc)
{
    const int block_m = 64; 
    const int block_n = 64;   
    const int block_k = 128;
    const int prefetch_dist = 64;
    const int MR = 3;
    const int NR = 4;

    for (int ti = 0; ti < M; ti += block_m)
    {
        const int i_end = (ti + block_m < M) ? ti + block_m : M;

        for (int tj = 0; tj < N; tj += block_n)
        {
            const int j_end = (tj + block_n < N) ? tj + block_n : N;

            int i = ti;
            for (; i + MR <= i_end; i += MR)
            {
                const float *a0 = A + static_cast<long>(i) * lda;
                const float *a1 = A + static_cast<long>(i + 1) * lda;
                const float *a2 = A + static_cast<long>(i + 2) * lda;
                float *c0 = C + static_cast<long>(i) * ldc;
                float *c1 = C + static_cast<long>(i + 1) * ldc;
                float *c2 = C + static_cast<long>(i + 2) * ldc;

                int j = tj;
                for (; j + NR <= j_end; j += NR)
                {
                    const float *b0 = B + static_cast<long>(j + 0) * ldb;
                    const float *b1 = B + static_cast<long>(j + 1) * ldb;
                    const float *b2 = B + static_cast<long>(j + 2) * ldb;
                    const float *b3 = B + static_cast<long>(j + 3) * ldb;
                    kernel_3x4(a0, a1, a2, b0, b1, b2, b3, K, block_k, prefetch_dist,
                               c0 + j, c1 + j, c2 + j);
                }

                for (; j < j_end; ++j)
                {
                    const float *b = B + static_cast<long>(j) * ldb;
                    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
                    for (int tk = 0; tk < K; tk += block_k)
                    {
                        const int tile_k = (tk + block_k < K) ? block_k : K - tk;
                        r0 += dot_masked(a0 + tk, b + tk, tile_k, prefetch_dist);
                        r1 += dot_masked(a1 + tk, b + tk, tile_k, prefetch_dist);
                        r2 += dot_masked(a2 + tk, b + tk, tile_k, prefetch_dist);
                    }
                    c0[j] = r0; c1[j] = r1; c2[j] = r2;
                }
            }

            for (; i < i_end; ++i)
            {
                const float *a = A + static_cast<long>(i) * lda;
                float *c = C + static_cast<long>(i) * ldc;
                for (int j = tj; j < j_end; ++j)
                {
                    const float *b = B + static_cast<long>(j) * ldb;
                    float r = 0.0f;
                    for (int tk = 0; tk < K; tk += block_k)
                    {
                        const int tile_k = (tk + block_k < K) ? block_k : K - tk;
                        r += dot_masked(a + tk, b + tk, tile_k, prefetch_dist);
                    }
                    c[j] = r;
                }
            }
        }
    }
}

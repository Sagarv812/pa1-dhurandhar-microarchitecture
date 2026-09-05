// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

static float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static float dot_simd(const float* a, const float* b, int K) {
    __m256 acc = _mm256_setzero_ps();
    int p = 0;
    for (; p + 8 <= K; p += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
    float s = hsum256(acc);
    for (; p < K; ++p) s += a[p] * b[p];
    return s;
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    const int step = 4;

    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;

        int j = 0;
        for (; j + step <= N; j += step) {
            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int p = 0;
            for (; p + 8 <= K; p += 8) {
                __m256 a_vals = _mm256_loadu_ps(a + p);
                acc0 = _mm256_fmadd_ps(a_vals, _mm256_loadu_ps(b0 + p), acc0);
                acc1 = _mm256_fmadd_ps(a_vals, _mm256_loadu_ps(b1 + p), acc1);
                acc2 = _mm256_fmadd_ps(a_vals, _mm256_loadu_ps(b2 + p), acc2);
                acc3 = _mm256_fmadd_ps(a_vals, _mm256_loadu_ps(b3 + p), acc3);
            }

            float s0 = hsum256(acc0), s1 = hsum256(acc1);
            float s2 = hsum256(acc2), s3 = hsum256(acc3);
            for (; p < K; ++p) {
                s0 += a[p] * b0[p];
                s1 += a[p] * b1[p];
                s2 += a[p] * b2[p];
                s3 += a[p] * b3[p];
            }

            float* c = C + static_cast<long>(i) * ldc + j;
            c[0] = s0; c[1] = s1; c[2] = s2; c[3] = s3;
        }

        for (; j < N; ++j)
            C[static_cast<long>(i) * ldc + j] = dot_simd(a, B + static_cast<long>(j) * ldb, K);
    }
}

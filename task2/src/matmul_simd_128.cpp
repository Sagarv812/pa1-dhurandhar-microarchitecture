// matmul_simd_128.cpp  STAGE 1 (128-bit variant, ungraded)
// Same logic as matmul_simd.cpp but with 128-bit registers (4 floats/vector)
// instead of 256-bit. Used only for the SIMD-width sweep. Build with -msse4.2 -mfma.
#include <immintrin.h>

#include "matmul.h"

static float hsum128(__m128 v) {
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

static float dot_simd(const float* a, const float* b, int K) {
    __m128 acc = _mm_setzero_ps();
    int p = 0;
    for (; p + 4 <= K; p += 4)
        acc = _mm_fmadd_ps(_mm_loadu_ps(a + p), _mm_loadu_ps(b + p), acc);
    float s = hsum128(acc);
    for (; p < K; ++p) s += a[p] * b[p];
    return s;
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    const int step = 4;

    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;

        int j = 0;
        for (; j + step <= N; j += step) {
            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps();
            __m128 acc3 = _mm_setzero_ps();

            int p = 0;
            for (; p + 4 <= K; p += 4) {
                __m128 a_vals = _mm_loadu_ps(a + p);
                acc0 = _mm_fmadd_ps(a_vals, _mm_loadu_ps(b0 + p), acc0);
                acc1 = _mm_fmadd_ps(a_vals, _mm_loadu_ps(b1 + p), acc1);
                acc2 = _mm_fmadd_ps(a_vals, _mm_loadu_ps(b2 + p), acc2);
                acc3 = _mm_fmadd_ps(a_vals, _mm_loadu_ps(b3 + p), acc3);
            }

            float s0 = hsum128(acc0), s1 = hsum128(acc1);
            float s2 = hsum128(acc2), s3 = hsum128(acc3);
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

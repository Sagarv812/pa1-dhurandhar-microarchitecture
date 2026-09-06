#include <immintrin.h>

#include "matmul.h"

static inline float horizontal_sum(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value),
                            _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline void compute_one_row(const float* a, float* c_row,
                                   const float* b0, const float* b1,
                                   const float* b2, const float* b3,
                                   int cols, int K, int column) {
    __m256 accumulators[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()};
    int p = 0;

    for (; p + 8 <= K; p += 8) {
        const __m256 va = _mm256_loadu_ps(a + p);
        accumulators[0] = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + p), accumulators[0]);
        if (cols > 1)
            accumulators[1] = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + p), accumulators[1]);
        if (cols > 2)
            accumulators[2] = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2 + p), accumulators[2]);
        if (cols > 3)
            accumulators[3] = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3 + p), accumulators[3]);
    }

    const float* b_rows[4] = {b0, b1, b2, b3};
    for (int col = 0; col < cols; ++col) {
        float sum = horizontal_sum(accumulators[col]);
        for (int tail = p; tail < K; ++tail)
            sum += a[tail] * b_rows[col][tail];
        c_row[column + col] = sum;
    }
}

static inline void compute_two_rows(const float* a0, const float* a1,
                                    float* c0, float* c1,
                                    const float* b0, const float* b1,
                                    const float* b2, const float* b3,
                                    int cols, int K, int column) {
    __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
    __m256 acc02 = _mm256_setzero_ps(), acc03 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
    __m256 acc12 = _mm256_setzero_ps(), acc13 = _mm256_setzero_ps();

    int p = 0;
    for (; p + 8 <= K; p += 8) {
        const __m256 va0 = _mm256_loadu_ps(a0 + p);
        const __m256 va1 = _mm256_loadu_ps(a1 + p);
        const __m256 vb0 = _mm256_loadu_ps(b0 + p);
        acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
        acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
        if (cols > 1) { const __m256 vb1 = _mm256_loadu_ps(b1 + p); acc01 = _mm256_fmadd_ps(va0, vb1, acc01); acc11 = _mm256_fmadd_ps(va1, vb1, acc11); }
        if (cols > 2) { const __m256 vb2 = _mm256_loadu_ps(b2 + p); acc02 = _mm256_fmadd_ps(va0, vb2, acc02); acc12 = _mm256_fmadd_ps(va1, vb2, acc12); }
        if (cols > 3) { const __m256 vb3 = _mm256_loadu_ps(b3 + p); acc03 = _mm256_fmadd_ps(va0, vb3, acc03); acc13 = _mm256_fmadd_ps(va1, vb3, acc13); }
    }

    __m256 accumulators0[4] = {acc00, acc01, acc02, acc03};
    __m256 accumulators1[4] = {acc10, acc11, acc12, acc13};
    const float* b_rows[4] = {b0, b1, b2, b3};
    for (int col = 0; col < cols; ++col) {
        float sum0 = horizontal_sum(accumulators0[col]);
        float sum1 = horizontal_sum(accumulators1[col]);
        for (int tail = p; tail < K; ++tail) {
            sum0 += a0[tail] * b_rows[col][tail];
            sum1 += a1[tail] * b_rows[col][tail];
        }
        c0[column + col] = sum0;
        c1[column + col] = sum1;
    }
}

void matmul_optimized(const float* __restrict A, const float* __restrict B, float* __restrict C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    const int T = 64;

    for (int jb = 0; jb < N; jb += T) {
        const int j_end = (jb + T < N) ? jb + T : N;
        for (int ib = 0; ib < M; ib += T) {
            const int i_end = (ib + T < M) ? ib + T : M;

            int i = ib;
            for (; i + 2 < i_end; i += 3) {
                const float* a0 = A + static_cast<long>(i) * lda;
                const float* a1 = A + static_cast<long>(i + 1) * lda;
                const float* a2 = A + static_cast<long>(i + 2) * lda;
                float* c0 = C + static_cast<long>(i) * ldc;
                float* c1 = C + static_cast<long>(i + 1) * ldc;
                float* c2 = C + static_cast<long>(i + 2) * ldc;

                for (int j = jb; j < j_end; j += 4) {
                    const int cols = (j_end - j < 4) ? j_end - j : 4;
                    const float* b0 = B + static_cast<long>(j) * ldb;
                    const float* b1 = (cols > 1) ? b0 + ldb : nullptr;
                    const float* b2 = (cols > 2) ? b0 + 2L * ldb : nullptr;
                    const float* b3 = (cols > 3) ? b0 + 3L * ldb : nullptr;

                    __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
                    __m256 acc02 = _mm256_setzero_ps(), acc03 = _mm256_setzero_ps();
                    __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
                    __m256 acc12 = _mm256_setzero_ps(), acc13 = _mm256_setzero_ps();
                    __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
                    __m256 acc22 = _mm256_setzero_ps(), acc23 = _mm256_setzero_ps();

                    int p = 0;
                    for (; p + 8 <= K; p += 8) {
                        const __m256 va0 = _mm256_loadu_ps(a0 + p);
                        const __m256 va1 = _mm256_loadu_ps(a1 + p);
                        const __m256 va2 = _mm256_loadu_ps(a2 + p);
                        const __m256 vb0 = _mm256_loadu_ps(b0 + p);
                        acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                        acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                        acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                        if (cols > 1) { const __m256 vb1 = _mm256_loadu_ps(b1 + p); acc01 = _mm256_fmadd_ps(va0, vb1, acc01); acc11 = _mm256_fmadd_ps(va1, vb1, acc11); acc21 = _mm256_fmadd_ps(va2, vb1, acc21); }
                        if (cols > 2) { const __m256 vb2 = _mm256_loadu_ps(b2 + p); acc02 = _mm256_fmadd_ps(va0, vb2, acc02); acc12 = _mm256_fmadd_ps(va1, vb2, acc12); acc22 = _mm256_fmadd_ps(va2, vb2, acc22); }
                        if (cols > 3) { const __m256 vb3 = _mm256_loadu_ps(b3 + p); acc03 = _mm256_fmadd_ps(va0, vb3, acc03); acc13 = _mm256_fmadd_ps(va1, vb3, acc13); acc23 = _mm256_fmadd_ps(va2, vb3, acc23); }
                    }

                    __m256 accumulators0[4] = {acc00, acc01, acc02, acc03};
                    __m256 accumulators1[4] = {acc10, acc11, acc12, acc13};
                    __m256 accumulators2[4] = {acc20, acc21, acc22, acc23};
                    const float* b_rows[4] = {b0, b1, b2, b3};
                    for (int col = 0; col < cols; ++col) {
                        float sum0 = horizontal_sum(accumulators0[col]);
                        float sum1 = horizontal_sum(accumulators1[col]);
                        float sum2 = horizontal_sum(accumulators2[col]);
                        for (int tail = p; tail < K; ++tail) {
                            sum0 += a0[tail] * b_rows[col][tail];
                            sum1 += a1[tail] * b_rows[col][tail];
                            sum2 += a2[tail] * b_rows[col][tail];
                        }
                        c0[j + col] = sum0;
                        c1[j + col] = sum1;
                        c2[j + col] = sum2;
                    }
                }
            }

            if (i + 1 < i_end) {
                const float* a0 = A + static_cast<long>(i) * lda;
                const float* a1 = A + static_cast<long>(i + 1) * lda;
                float* c0 = C + static_cast<long>(i) * ldc;
                float* c1 = C + static_cast<long>(i + 1) * ldc;
                for (int j = jb; j < j_end; j += 4) {
                    const int cols = (j_end - j < 4) ? j_end - j : 4;
                    const float* b0 = B + static_cast<long>(j) * ldb;
                    compute_two_rows(a0, a1, c0, c1, b0,
                                     (cols > 1) ? b0 + ldb : nullptr,
                                     (cols > 2) ? b0 + 2L * ldb : nullptr,
                                     (cols > 3) ? b0 + 3L * ldb : nullptr,
                                     cols, K, j);
                }
                i += 2;
            } else if (i < i_end) {
                const float* a = A + static_cast<long>(i) * lda;
                float* c_row = C + static_cast<long>(i) * ldc;
                for (int j = jb; j < j_end; j += 4) {
                    const int cols = (j_end - j < 4) ? j_end - j : 4;
                    const float* b0 = B + static_cast<long>(j) * ldb;
                    compute_one_row(a, c_row, b0,
                                    (cols > 1) ? b0 + ldb : nullptr,
                                    (cols > 2) ? b0 + 2L * ldb : nullptr,
                                    (cols > 3) ? b0 + 3L * ldb : nullptr,
                                    cols, K, j);
                }
                i += 1;
            }
        }
    }
}

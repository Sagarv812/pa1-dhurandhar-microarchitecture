
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

static float hsum256(__m256 v){
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static float dot_prefetch(const float* a, const float* b, int length, int prefetch_dist){
    int p = 0;
    __m256 acc = _mm256_setzero_ps();
    for(; p + 8 <= length ; p+=8){
        if(p+ prefetch_dist < length){
            _mm_prefetch(reinterpret_cast<const char*>(a + p + prefetch_dist) , _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b + p + prefetch_dist) , _MM_HINT_T0);
        }
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+p) , _mm256_loadu_ps(b+p), acc);
    }
    float result = hsum256(acc);
    for(; p < length ; p++){
        result += a[p] * b[p];
    }
    return result;
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int block_m = 32;
    const int block_n = 32;
    const int block_k = 128;
    const int prefetch_dist = 16;
    const int block_out = 4;
    for(int ti = 0; ti < M ; ti += block_m){
        const int i_end = (ti + block_m < M) ? ti + block_m : M;

        for(int tj = 0 ; tj < N ; tj+= block_n){
            const int j_end = (tj + block_n < N) ? tj + block_n : N;

            for (int i = ti; i < i_end; ++i) {
                const float* a_base = A + static_cast<long>(i) * lda;
                float* c_row = C + static_cast<long>(i) * ldc;

                int j = tj;
                for (; j + block_out <= j_end; j += block_out) {
                    const float* b0 = B + static_cast<long>(j + 0) * ldb;
                    const float* b1 = B + static_cast<long>(j + 1) * ldb;
                    const float* b2 = B + static_cast<long>(j + 2) * ldb;
                    const float* b3 = B + static_cast<long>(j + 3) * ldb;

                    // Keep the output accumulators alive across all K tiles.
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    float tail0 = 0.0f;
                    float tail1 = 0.0f;
                    float tail2 = 0.0f;
                    float tail3 = 0.0f;

                    for (int tk = 0; tk < K; tk += block_k) {
                        const int tile_k =
                            (tk + block_k < K) ? block_k : K - tk;
                        const float* a = a_base + tk;
                        const float* current_b0 = b0 + tk;
                        const float* current_b1 = b1 + tk;
                        const float* current_b2 = b2 + tk;
                        const float* current_b3 = b3 + tk;

                        int p = 0;
                        for (; p + 16 <= tile_k; p += 16) {
                            if (p + prefetch_dist < tile_k) {
                                _mm_prefetch(reinterpret_cast<const char*>(a + p + prefetch_dist), _MM_HINT_T0);
                                _mm_prefetch(reinterpret_cast<const char*>(current_b0 + p + prefetch_dist), _MM_HINT_T0);
                                _mm_prefetch(reinterpret_cast<const char*>(current_b1 + p + prefetch_dist), _MM_HINT_T0);
                                _mm_prefetch(reinterpret_cast<const char*>(current_b2 + p + prefetch_dist), _MM_HINT_T0);
                                _mm_prefetch(reinterpret_cast<const char*>(current_b3 + p + prefetch_dist), _MM_HINT_T0);
                            }

                            __m256 a0 = _mm256_loadu_ps(a + p);
                            __m256 a1 = _mm256_loadu_ps(a + p + 8);

                            acc0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(current_b0 + p), acc0);
                            acc1 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(current_b1 + p), acc1);
                            acc2 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(current_b2 + p), acc2);
                            acc3 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(current_b3 + p), acc3);
                            acc0 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(current_b0 + p + 8), acc0);
                            acc1 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(current_b1 + p + 8), acc1);
                            acc2 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(current_b2 + p + 8), acc2);
                            acc3 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(current_b3 + p + 8), acc3);
                        }

                        for (; p + 8 <= tile_k; p += 8) {
                            if (p + prefetch_dist < tile_k) {
                                _mm_prefetch(reinterpret_cast<const char*>(a + p + prefetch_dist), _MM_HINT_T0);
                                _mm_prefetch(reinterpret_cast<const char*>(current_b0 + p + prefetch_dist), _MM_HINT_T0);
                                _mm_prefetch(reinterpret_cast<const char*>(current_b1 + p + prefetch_dist), _MM_HINT_T0);
                                _mm_prefetch(reinterpret_cast<const char*>(current_b2 + p + prefetch_dist), _MM_HINT_T0);
                                _mm_prefetch(reinterpret_cast<const char*>(current_b3 + p + prefetch_dist), _MM_HINT_T0);
                            }

                            __m256 a_values = _mm256_loadu_ps(a + p);
                            acc0 = _mm256_fmadd_ps(a_values, _mm256_loadu_ps(current_b0 + p), acc0);
                            acc1 = _mm256_fmadd_ps(a_values, _mm256_loadu_ps(current_b1 + p), acc1);
                            acc2 = _mm256_fmadd_ps(a_values, _mm256_loadu_ps(current_b2 + p), acc2);
                            acc3 = _mm256_fmadd_ps(a_values, _mm256_loadu_ps(current_b3 + p), acc3);
                        }

                        for (; p < tile_k; ++p) {
                            tail0 += a[p] * current_b0[p];
                            tail1 += a[p] * current_b1[p];
                            tail2 += a[p] * current_b2[p];
                            tail3 += a[p] * current_b3[p];
                        }
                    }

                    c_row[j + 0] = hsum256(acc0) + tail0;
                    c_row[j + 1] = hsum256(acc1) + tail1;
                    c_row[j + 2] = hsum256(acc2) + tail2;
                    c_row[j + 3] = hsum256(acc3) + tail3;
                }

                for (; j < j_end; ++j) {
                    float result = 0.0f;
                    const float* b = B + static_cast<long>(j) * ldb;

                    for (int tk = 0; tk < K; tk += block_k) {
                        const int tile_k =
                            (tk + block_k < K) ? block_k : K - tk;
                        result += dot_prefetch(
                            a_base + tk, b + tk, tile_k, prefetch_dist);
                    }

                    c_row[j] = result;
                }
            }
        }
    }
}
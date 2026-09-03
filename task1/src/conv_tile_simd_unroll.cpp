// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_tile_simd_unroll(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int T = 32;
    const int step = 8;

    for (int ty = 0; ty < H; ty += T) {
        const int oy_end = (ty + T < H) ? ty + T : H;
        for (int tx = 0; tx < W; tx += T) {
            const int ox_end = (tx + T < W) ? tx + T : W;

            for (int oy = ty; oy < oy_end; oy++) {
                for (int ox = tx; ox < ox_end; ox+=4*step) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            __m256 ker_val = _mm256_set1_ps(ker[ky*K + kx]);
                            const float* in_ptr = in + (oy + ky) * in_stride + (ox + kx);

                            __m256 in0 = _mm256_loadu_ps(in_ptr);
                            __m256 in1 = _mm256_loadu_ps(in_ptr + 8);
                            __m256 in2 = _mm256_loadu_ps(in_ptr + 16);
                            __m256 in3 = _mm256_loadu_ps(in_ptr + 24);

                            acc0 = _mm256_fmadd_ps(ker_val, in0, acc0);
                            acc1 = _mm256_fmadd_ps(ker_val, in1, acc1);
                            acc2 = _mm256_fmadd_ps(ker_val, in2, acc2);
                            acc3 = _mm256_fmadd_ps(ker_val, in3, acc3);
                        }
                    }

                    _mm256_storeu_ps(out + oy*W + ox,      acc0);
                    _mm256_storeu_ps(out + oy*W + ox + 8,  acc1);
                    _mm256_storeu_ps(out + oy*W + ox + 16, acc2);
                    _mm256_storeu_ps(out + oy*W + ox + 24, acc3);
                }
            }
        }
    }
}

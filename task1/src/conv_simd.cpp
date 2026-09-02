// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int step = 8;

    for (int oy = 0; oy < H; oy++) {
        for (int ox = 0; ox < W; ox+=step) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ky++) {
                for (int kx = 0; kx < K; kx++) {
                    __m256 ker_val = _mm256_set1_ps(ker[ky*K + kx]);
                    const float* in_row = in + (oy + ky) * in_stride + (ox + kx);
                    __m256 in_vals = _mm256_loadu_ps(in_row);
                    acc = _mm256_fmadd_ps(ker_val, in_vals, acc);
                }
            }
            float *out_row = out + oy * W + ox;
            _mm256_storeu_ps(out_row, acc);
        }
    }
    // conv_naive(in, out, ker, H, W, K);
}

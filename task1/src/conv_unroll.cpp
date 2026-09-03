// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int step = 4;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += step) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    float k = ker[ky * K + kx];
                    acc0 += in_row[kx]     * k;
                    acc1 += in_row[kx + 1] * k;
                    acc2 += in_row[kx + 2] * k;
                    acc3 += in_row[kx + 3] * k;
                }
            }
            out[oy * W + ox]     = acc0;
            out[oy * W + ox + 1] = acc1;
            out[oy * W + ox + 2] = acc2;
            out[oy * W + ox + 3] = acc3;
        }
    }
}

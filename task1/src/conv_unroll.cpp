// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            out[oy*W + ox] = 0.0f;
        }
    }

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            float ker_val = ker[ky * K + kx];
            for (int oy = 0; oy < H; ++oy) {
                const float* in_row = in + (oy + ky) * in_stride + kx;
                float* out_row = out + oy * W;
                for (int ox = 0; ox < W; ox += 4) {
                    out_row[ox] += in_row[ox] * ker_val;
                    out_row[ox+1] += in_row[ox+1] * ker_val;
                    out_row[ox+2] += in_row[ox+2] * ker_val;
                    out_row[ox+3] += in_row[ox+3] * ker_val;
                }
            }
        }
    }
    // conv_naive(in, out, ker, H, W, K);
}

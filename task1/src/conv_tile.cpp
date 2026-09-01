// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int T = 64;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            out[oy * W + ox] = 0.0f;
        }
    }

    for (int ty = 0; ty < H; ty += T) {
        const int oy_end = (ty + T < H) ? ty + T : H;
        for (int tx = 0; tx < W; tx += T) {
            const int ox_end = (tx + T < W) ? tx + T : W;

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float ker_val = ker[ky * K + kx];
                    for (int oy = ty; oy < oy_end; ++oy) {
                        const float* in_row = in + (oy + ky) * in_stride + kx;
                        float* out_row = out + oy * W;
                        for (int ox = tx; ox < ox_end; ++ox) {
                            out_row[ox] += in_row[ox] * ker_val;
                        }
                    }
                }
            }
        }
    }
}

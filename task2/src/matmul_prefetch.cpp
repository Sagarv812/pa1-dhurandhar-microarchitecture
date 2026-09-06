// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

#define PREFETCH_HINT _MM_HINT_T0

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    const int prefetch_dist = 64;
    const int T = 64;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[static_cast<long>(i) * ldc + j] = 0.0f;
        }
    }

    for (int io = 0; io < M; io += T) {
        int i_end = (io + T < M)? io + T : M;
        for (int jo = 0; jo < N; jo += T) {
            int j_end = (jo + T < N)? jo + T : N;
            for (int ko = 0; ko < K; ko += T) {
                
                int k_end = (ko + T < K)? ko + T : K;

                for(int i = io; i < i_end; i++) {
                    for (int j = jo; j < j_end; j++) {
                        float acc = 0.0f;
                        const float* a = A + static_cast<long>(i)*lda;
                        const float* b = B + static_cast<long>(j)*ldb;

                        int p = ko;

                        for(; p + 16 <= k_end; p+=16) {
                            if (p + prefetch_dist < K) {
                                _mm_prefetch(&a[p+prefetch_dist], PREFETCH_HINT);
                                _mm_prefetch(&b[p+prefetch_dist], PREFETCH_HINT);
                            }

                            for(int step = 0; step < 16; step++){
                                acc += a[p+step]*b[p+step];
                            }
                        }

                        for(; p < k_end; p++){
                            acc += a[p]*b[p];
                        }

                        C[static_cast<long>(i)*ldc + j] += acc;
                    }
                }
            }
        }
    }

}

// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    const int PREFETCH_DISTANCE = 16;
    const int T = 64;

    for (int i = 0 ; i < M ; i++){
        for (int j = 0; j <  N ; j++){
            C[static_cast<long>(i) * ldc + j] = 0.0f ;
        }
    }

    for (int jb = 0 ; jb < N ;jb += T){
        const int j_end = (jb + T < N) ? jb + T : N;
        for (int ib = 0 ; ib < M ; ib += T){
            const int i_end = (ib + T < M) ?  ib + T : M;
            //in block i,j
            for (int kb = 0 ; kb < K ; kb += T){
                const int k_end = (kb + T < K) ?  kb + T : K;
            
        
            for (int i = ib; i < i_end; ++i) {
                const float* a = A + static_cast<long>(i) * lda;

            for (int j = jb; j < j_end; ++j) {
                float acc = 0.0f;
                const float* b = B + static_cast<long>(j) * ldb;
                for (int p = kb; p < k_end; p+= 16) {
                    _mm_prefetch(reinterpret_cast<const char*>(&a[p + PREFETCH_DISTANCE]), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(&b[p + PREFETCH_DISTANCE]), _MM_HINT_T0);
                    acc += a[p] * b[p];
                    acc += a[p + 1] * b[p+1];
                    acc += a[p + 2] * b[p + 2];
                    acc += a[p + 3] * b[p + 3];
                    acc += a[p + 4] * b[p + 4];
                    acc += a[p + 5] * b[p + 5];
                    acc += a[p + 6] * b[p + 6];
                    acc += a[p + 7] * b[p + 7];
                    acc += a[p + 8] * b[p + 8];
                    acc += a[p + 9] * b[p + 9];
                    acc += a[p + 10] * b[p + 10];
                    acc += a[p + 11] * b[p + 11];
                    acc += a[p + 12] * b[p + 12];
                    acc += a[p + 13] * b[p + 13];
                    acc += a[p + 14] * b[p + 14];
                    acc += a[p + 15] * b[p + 15];
                    
                }
                C[static_cast<long>(i) * ldc + j] += acc;
            }
        }
    }


        }
    }


}

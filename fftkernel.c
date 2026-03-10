#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#define RUNS 100000000
#define CPU_FREQ_GHZ 3.40

#define SIMD_ADD(dest, s1, s2) \
__asm__ __volatile__ (      \
  "vaddpd %[rs1], %[rs2], %[rdest]\n" \
  : [rdest] "+x"(dest)      \
  : [rs1] "x"(s1), [rs2] "x"(s2)         \
);

#define SIMD_SUB(dest, s1, s2) \
__asm__ __volatile__ (      \
  "vsubpd %[rs1], %[rs2], %[rdest]\n" \
  : [rdest] "+x"(dest)      \
  : [rs1] "x"(s1), [rs2] "x"(s2)         \
);

// Timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

/* * 4-point FFT Kernel (Structure of Arrays)
 * Operation Count Analysis:
 * -------------------------
 * This kernel performs 16 AVX arithmetic instructions (vaddpd/vsubpd).
 * Each AVX instruction operates on 4 doubles.
 * Total FLOPs per call = 16 instrs * 4 doubles = 64 FLOPs.
 */
__inline__ void fft4_kernel(const double *in_re, const double *in_im, 
                 double *out_re, double *out_im, int size) 
{
    for (int i = 0; i < size; i += 16) {
        register __m256d R0 = _mm256_load_pd(in_re);   // X0 real
        in_re += 8;
        register __m256d R2 = _mm256_load_pd(in_re);   // X2 real
        in_re -= 4;
        register __m256d Ar;
        SIMD_ADD(Ar, R0, R2);
        register __m256d R1 = _mm256_load_pd(in_re);   // X1 real
        in_re += 8;
        register __m256d R3 = _mm256_load_pd(in_re);  // X3 real
        in_re += 4;
        register __m256d Cr;
        SIMD_ADD(Cr, R1, R3);

        register __m256d R4 = _mm256_load_pd(in_im);   // X0 im
        in_im += 8;
        register __m256d R6 = _mm256_load_pd(in_im);   // X2 im
        in_im -= 4;
        register __m256d Ai;
        SIMD_ADD(Ai, R4, R6);
        register __m256d R5 = _mm256_load_pd(in_im);   // X1 im
        in_im += 8;
        register __m256d R7 = _mm256_load_pd(in_im);  // X3 im
        in_im += 4;
        register __m256d Ci;
        SIMD_ADD(Ci, R5, R7);

        register __m256d Br;
        SIMD_SUB(Br, R2, R0);
        register __m256d Bi;
        SIMD_SUB(Bi, R6, R4);
        register __m256d Dr;
        SIMD_SUB(Dr, R3, R1);
        register __m256d Di;
        SIMD_SUB(Di, R7, R5);

        SIMD_ADD(R0, Ar, Cr); // F0 real
        _mm256_store_pd(out_re, R0);
        out_re += 4;
        SIMD_ADD(R1, Ai, Ci); // F0 im
        _mm256_store_pd(out_im, R1);
        out_im += 4;

        SIMD_ADD(R2, Br, Di); // F1 real
        _mm256_store_pd(out_re, R2);
        out_re += 4;
        SIMD_SUB(R3, Dr, Bi); // F1 im
        _mm256_store_pd(out_im, R3);
        out_im += 4;

        SIMD_SUB(R4, Cr, Ar); // F2 real
        _mm256_store_pd(out_re, R4);
        out_re += 4;
        SIMD_SUB(R5, Ci, Ai); // F2 im
        _mm256_store_pd(out_im, R5);
        out_im += 4;

        SIMD_SUB(R6, Di, Br); // F3 real
        _mm256_store_pd(out_re, R6);
        out_re += 4;
        SIMD_ADD(R7, Dr, Bi); // F3 im
        _mm256_store_pd(out_im, R7);
        out_im += 4;
    }
}

int main() {
    // 1. Setup Memory
    size_t n_doubles = 128;
    size_t size = n_doubles * sizeof(double);
    
    double *in_re = (double*)_mm_malloc(size, 32);
    double *in_im = (double*)_mm_malloc(size, 32);
    double *out_re = (double*)_mm_malloc(size, 32);
    double *out_im = (double*)_mm_malloc(size, 32);

    // 2. Initialize Data
    for(int i=0; i<32; i++) { 
        for(int j=0; j<4; j++) { 
            int idx = i*4 + j;
            in_re[idx] = i+1; in_im[idx] = 0.0;
        }
    }

    // 3. Warm Up (ensure instructions/data are in cache)
    for(int i=0; i<100000; i++) {
        fft4_kernel(in_re, in_im, out_re, out_im, n_doubles);
    }

    unsigned long long start = rdtsc();
    
    for(int i=0; i<RUNS; i+=4) {
        fft4_kernel(in_re, in_im, out_re, out_im, n_doubles);
        fft4_kernel(in_re, in_im, out_re, out_im, n_doubles);
        fft4_kernel(in_re, in_im, out_re, out_im, n_doubles);
        fft4_kernel(in_re, in_im, out_re, out_im, n_doubles);
    }
    
    unsigned long long end = rdtsc();

    unsigned long long total_cycles = end - start;
    double avg_cycles = ((double)total_cycles) / RUNS;
    double seconds = (double)total_cycles / (CPU_FREQ_GHZ * 1e9);
    
    double total_flops = 64.0 * (n_doubles/16) * RUNS;
    double flopspercycle = (total_flops / total_cycles);

    // 6. Report
    printf("--- Performance Report ---\n");
    printf("CPU Frequency:  %.2f GHz\n", CPU_FREQ_GHZ);
    printf("Total Runs:     %d\n", RUNS);
    printf("Total Cycles:   %llu\n", total_cycles);
    printf("Avg Cycles/Run: %.2f\n", avg_cycles);
    printf("Est. Time (s):  %.6f\n", seconds);
    printf("Throughput:     %.2f FLOPS PER CYCLE\n", flopspercycle);
    printf("--------------------------\n");

    // 7. Verify Accuracy (Instance 0)
    for(int k=0; k<4; k++) {
        printf("F[%d]: %.2f + %.2fi\n", k, out_re[k*4], out_im[k*4]);
    }

    _mm_free(in_re); _mm_free(in_im);
    _mm_free(out_re); _mm_free(out_im);

    return 0;
}
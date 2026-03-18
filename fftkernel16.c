#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>

#define RUNS 10000000
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

#define SIMD_MUL(dest, s1, s2) \
__asm__ __volatile__ (      \
  "vmulpd %[rs1], %[rs2], %[rdest]\n" \
  : [rdest] "+x"(dest)      \
  : [rs1] "x"(s1), [rs2] "x"(s2)         \
);

#define SIMD_FMA(dest, s1, s2)                             \
__asm__ __volatile__(                                    \
  "vfmadd231pd %[rs1], %[rs2], %[rd]\n\t" /* d += s1*s2 */ \
  : [rd] "+x"(dest)                                      \
  : [rs1] "x"(s1), [rs2] "x"(s2)                         \
);

#define SIMD_FMS(dest, s1, s2)                             \
__asm__ __volatile__(                                    \
  "vfnmadd231pd %[rs1], %[rs2], %[rd]\n\t" /* d += s1*s2 */ \
  : [rd] "+x"(dest)                                      \
  : [rs1] "x"(s1), [rs2] "x"(s2)                         \
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

__inline__ void fft4_kernel_transpose(const double *in_re, const double *in_im, 
                 double *out_re, double *out_im, int size, const double* twiddle_r, const double* twiddle_i) 
{
    for (int i = 0; i < size; i += 16) {
        register __m256d R0 = _mm256_load_pd(in_re);   // X0 real
        in_re += 8;
        register __m256d R2 = _mm256_load_pd(in_re);   // X2 real
        in_re -= 4;
        register __m256d R8;
        SIMD_ADD(R8, R0, R2);
        register __m256d R1 = _mm256_load_pd(in_re);   // X1 real
        in_re += 8;
        register __m256d R3 = _mm256_load_pd(in_re);  // X3 real
        in_re += 4;
        register __m256d R9;
        SIMD_ADD(R9, R1, R3);

        register __m256d R4 = _mm256_load_pd(in_im);   // X0 im
        in_im += 8;
        register __m256d R6 = _mm256_load_pd(in_im);   // X2 im
        in_im -= 4;
        register __m256d R10;
        SIMD_ADD(R10, R4, R6);
        register __m256d R5 = _mm256_load_pd(in_im);   // X1 im
        in_im += 8;
        register __m256d R7 = _mm256_load_pd(in_im);  // X3 im
        in_im += 4;
        register __m256d R11;
        SIMD_ADD(R11, R5, R7);

        register __m256d R12;
        SIMD_SUB(R12, R2, R0);
        register __m256d R13;
        SIMD_SUB(R13, R6, R4);
        register __m256d R14;
        SIMD_SUB(R14, R3, R1);
        register __m256d R15;
        SIMD_SUB(R15, R7, R5);

        SIMD_ADD(R0, R8, R9); // F0 real
        SIMD_ADD(R1, R10, R11); // F0 im

        SIMD_ADD(R2, R12, R15); // F1 real
        SIMD_SUB(R3, R14, R13); // F1 im

        SIMD_SUB(R4, R9, R8); // F2 real
        SIMD_SUB(R5, R11, R10); // F2 im

        SIMD_SUB(R6, R15, R12); // F3 real
        SIMD_ADD(R7, R14, R13); // F3 im
        
        //Transpose twiddle begin
        R9 = _mm256_load_pd(twiddle_r);
        twiddle_r += 4;
        R10 = _mm256_load_pd(twiddle_i);
        twiddle_i += 4;
        SIMD_MUL(R8, R0, R10); //ad
        SIMD_MUL(R0, R0, R9); //ac
        SIMD_FMS(R0, R1, R10) //ac-bd F0 real
        SIMD_FMA(R8, R1, R9); //ad+bc F0 im

        R10 = _mm256_load_pd(twiddle_r);
        twiddle_r += 4;
        R11 = _mm256_load_pd(twiddle_i);
        twiddle_i += 4;
        SIMD_MUL(R9, R2, R11); //ad
        SIMD_MUL(R1, R2, R10); //ac
        SIMD_FMS(R1, R3, R11); //ac-bd F1 real
        SIMD_FMA(R9, R3, R10); //ad+bc F1 im

        R12 = _mm256_load_pd(twiddle_r);
        twiddle_r += 4;
        R13 = _mm256_load_pd(twiddle_i);
        twiddle_i += 4;
        SIMD_MUL(R10, R4, R13); //ad
        SIMD_MUL(R2, R4, R12); //ac
        SIMD_FMS(R2, R5, R13); //ac-bd F2 real
        R4 = _mm256_permute2f128_pd(R0, R2, 0x20); // [A0 B0 C0 D0]
        SIMD_FMA(R10, R5, R12); //ad+bc F2 im
        R5 = _mm256_permute2f128_pd(R0, R2, 0x31); // [A1 B1 C1 D1]

        R13 = _mm256_load_pd(twiddle_r);
        twiddle_r -= 12;
        R14 = _mm256_load_pd(twiddle_i);
        twiddle_i -= 12;
        SIMD_MUL(R11, R6, R14); //ad
        SIMD_MUL(R3, R6, R13); //ac
        SIMD_FMS(R3, R7, R14); //ac-bd F3 real
        R6 = _mm256_permute2f128_pd(R1, R3, 0x20); // [A2 B2 C2 D2]
        SIMD_FMA(R11, R7, R13); //ad+bc F3 im
        R7 = _mm256_permute2f128_pd(R1, R3, 0x31); // [A3 B3 C3 D3]

        R12 = _mm256_permute2f128_pd(R8, R10, 0x20); // [A0 B0 C0 D0]
        R13 = _mm256_permute2f128_pd(R8, R10, 0x31); // [A1 B1 C1 D1]
        R14 = _mm256_permute2f128_pd(R9, R11, 0x20); // [A2 B2 C2 D2]
        R15 = _mm256_permute2f128_pd(R9, R11, 0x31); // [A3 B3 C3 D3]
        
        R0 = _mm256_shuffle_pd(R4, R6, 0x0); // [A0 B0 A2 B2]
        R1 = _mm256_shuffle_pd(R4, R6, 0xF); // [A1 B1 A3 B3]
        R2 = _mm256_shuffle_pd(R5, R7, 0x0); // [C0 D0 C2 D2]
        R3 = _mm256_shuffle_pd(R5, R7, 0xF); // [C1 D1 C3 D3]
                
        R8 = _mm256_shuffle_pd(R12, R14, 0x0); // [A0 B0 A2 B2]
        R9 = _mm256_shuffle_pd(R12, R14, 0xF); // [A1 B1 A3 B3]
        R10 = _mm256_shuffle_pd(R13, R15, 0x0); // [C0 D0 C2 D2]
        R11 = _mm256_shuffle_pd(R13, R15, 0xF); // [C1 D1 C3 D3]

        //Transpose twiddle end

        _mm256_store_pd(out_re, R0);
        out_re += 4;
        _mm256_store_pd(out_im, R8);
        out_im += 4;
        _mm256_store_pd(out_re, R1);
        out_re += 4;
        _mm256_store_pd(out_im, R9);
        out_im += 4;
        _mm256_store_pd(out_re, R2);
        out_re += 4;
        _mm256_store_pd(out_im, R10);
        out_im += 4;
        _mm256_store_pd(out_re, R3);
        out_re += 4;
        _mm256_store_pd(out_im, R11);
        out_im += 4;
    }
}

int main() {
    // 1. Setup Memory
    size_t n_doubles = 128;
    size_t size = n_doubles * sizeof(double);
    
    double *in_re = (double*)_mm_malloc(size, 32);
    double *in_im = (double*)_mm_malloc(size, 32);
    double *mid_re = (double*)_mm_malloc(size, 32);
    double *mid_im = (double*)_mm_malloc(size, 32);
    double *out_re = (double*)_mm_malloc(size, 32);
    double *out_im = (double*)_mm_malloc(size, 32);

    __m256d t1 = _mm256_set1_pd(1.0);
    __m256d t2 = _mm256_set1_pd(3.0);
    __m256d t3 = _mm256_set1_pd(4.0);
    SIMD_FMS(t1, t2, t3);
    double values[4];
    _mm256_store_pd(values, t1); // "u" for unaligned, safe for any stack array
    printf("[%f, %f, %f, %f]\n", values[0], values[1], values[2], values[3]);
    // 2. Initialize Data
    for(int i=0; i<32; i++) { 
        for(int j=0; j<4; j++) { 
            int idx = i*4 + j;
            in_re[idx] = idx; in_im[idx] = idx;
        }
    }

    double master_r[16], master_i[16];
    double pi = 3.14159265358979323846;
    for(int k=0; k<16; k++) {
         double theta = -2.0 * pi * k / 16.0;
         master_r[k] = cos(theta); // Requires #include <math.h>
         master_i[k] = sin(theta);
    }

    // 2. Define the Index Map for the 4x4 Decomposition
    // We need W^(row * col). 
    int indices[16] = {
        0, 0, 0, 0,
        0, 1, 2, 3,
        0, 2, 4, 6,
        0, 3, 6, 9
    };

    double twiddle_r[16] __attribute__((aligned(32)));
    double twiddle_i[16] __attribute__((aligned(32)));

    for(int i=0; i<16; i++) {
         int idx = indices[i];
         twiddle_r[i] = master_r[idx];
         twiddle_i[i] = master_i[idx];
    }

    unsigned long long start = rdtsc();
    
    for(int i=0; i<RUNS; i++) {
        fft4_kernel_transpose(in_re, in_im, mid_re, mid_im, n_doubles, twiddle_r, twiddle_i);
        fft4_kernel(mid_re, mid_im, out_re, out_im, n_doubles);
    }
    
    unsigned long long end = rdtsc();

    unsigned long long total_cycles = end - start;
    double avg_cycles = ((double)total_cycles) / RUNS;
    double seconds = (double)total_cycles / (CPU_FREQ_GHZ * 1e9);
    
    double total_flops = (64.0+160) * (n_doubles/16) * RUNS;
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
    for(int k=0; k<16; k++) {
        printf("F[%d]: %.2f + %.2fi\n", k, out_re[k], out_im[k]);
    }

    _mm_free(in_re); _mm_free(in_im);
    _mm_free(out_re); _mm_free(out_im);

    return 0;
}
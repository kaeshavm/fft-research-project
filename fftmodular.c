#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>

#define RUNS 20000000
#define CPU_FREQ_GHZ 3.40

#define ADD_DOUBLE_PTR(ptr, offset) \
__asm__ __volatile__ ( \
  "add %[bytes], %[rptr]\n" \
  : [rptr] "+r" (ptr) \
  : [bytes] "r" ((size_t)((offset) * 8)) \
);

#define SUB_DOUBLE_PTR(ptr, offset) \
__asm__ __volatile__ ( \
  "sub %[bytes], %[rptr]\n" \
  : [rptr] "+r" (ptr) \
  : [bytes] "r" ((size_t)((offset) * 8)) \
);

#define SIMD_ADD(dest, s1, s2) \
__asm__ __volatile__ (      \
  "vaddpd %[rs1], %[rs2], %[rdest]\n" \
  : [rdest] "=x"(dest)      \
  : [rs1] "x"(s1), [rs2] "x"(s2)         \
);

#define SIMD_SUB(dest, s1, s2) \
__asm__ __volatile__ (      \
  "vsubpd %[rs1], %[rs2], %[rdest]\n" \
  : [rdest] "=x"(dest)      \
  : [rs1] "x"(s1), [rs2] "x"(s2)         \
);

#define SIMD_MUL(dest, s1, s2) \
__asm__ __volatile__ (      \
  "vmulpd %[rs1], %[rs2], %[rdest]\n" \
  : [rdest] "=x"(dest)      \
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

#define PROCESS_TWIDDLES_START(stage, R1, R2, tr, ti, offset, iters) \
    if ((stage) == 0) { \
        (R1) = _mm256_load_pd(tr); \
        (tr) += (offset); \
        (R2) = _mm256_load_pd(ti); \
        (ti) += (offset); \
    } else { \
        (R1) = _mm256_broadcast_sd(tr); \
        (tr) += (iters); \
        (R2) = _mm256_broadcast_sd(ti); \
        (ti) += (iters); \
    } \

#define PROCESS_TWIDDLES_END(stage, R1, R2, tr, ti, offset, iters) \
    if ((stage) == 0) { \
        (R1) = _mm256_load_pd(tr); \
        (tr) -= (offset) * 3 - 4; \
        (R2) = _mm256_load_pd(ti); \
        (ti) -= (offset) * 3 - 4; \
    } else { \
        (R1) = _mm256_broadcast_sd(tr); \
        (tr) += (iters); \
        (R2) = _mm256_broadcast_sd(ti); \
        (ti) += (iters); \
    } \

// Timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

/* * 4-point FFT Kernel
 * This kernel performs 16 AVX arithmetic instructions (vaddpd/vsubpd).
 * Each AVX instruction operates on 4 doubles.
 * Total FLOPs per call = 16 instrs * 4 doubles = 64 FLOPs.
 */

__inline__ void fft4_kernel(double *in_re, double *in_im, 
                 double *out_re, double *out_im, int n_doubles, int fft_size, 
                 int read_offset, int write_offset) 
{   
    for (int i = 0; i < n_doubles/fft_size; i++) {
        int iters = fft_size/16;
        for (int j = 0; j < iters; j++) {
            register __m256d R0 = _mm256_load_pd(in_re);   // X0 real
            ADD_DOUBLE_PTR(in_re, read_offset*2);
            register __m256d R2 = _mm256_load_pd(in_re);   // X2 real
            SUB_DOUBLE_PTR(in_re, read_offset);
            register __m256d Ar;
            SIMD_ADD(Ar, R0, R2);
            register __m256d R1 = _mm256_load_pd(in_re);   // X1 real
            ADD_DOUBLE_PTR(in_re, read_offset*2);
            register __m256d R3 = _mm256_load_pd(in_re);  // X3 real
            SUB_DOUBLE_PTR(in_re, read_offset*3-4);
            register __m256d Cr;
            SIMD_ADD(Cr, R1, R3);

            register __m256d R4 = _mm256_load_pd(in_im);   // X0 im
            ADD_DOUBLE_PTR(in_im, read_offset*2);
            register __m256d R6 = _mm256_load_pd(in_im);   // X2 im
            SUB_DOUBLE_PTR(in_im, read_offset);
            register __m256d Ai;
            SIMD_ADD(Ai, R4, R6);
            register __m256d R5 = _mm256_load_pd(in_im);   // X1 im
            ADD_DOUBLE_PTR(in_im, read_offset*2);
            register __m256d R7 = _mm256_load_pd(in_im);  // X3 im
            SUB_DOUBLE_PTR(in_im, read_offset*3-4);
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
            ADD_DOUBLE_PTR(out_re, write_offset);
            SIMD_ADD(R1, Ai, Ci); // F0 im
            _mm256_store_pd(out_im, R1);
            ADD_DOUBLE_PTR(out_im, write_offset);

            SIMD_ADD(R2, Br, Di); // F1 real
            _mm256_store_pd(out_re, R2);
            ADD_DOUBLE_PTR(out_re, write_offset);
            SIMD_SUB(R3, Dr, Bi); // F1 im
            _mm256_store_pd(out_im, R3);
            ADD_DOUBLE_PTR(out_im, write_offset);

            SIMD_SUB(R4, Cr, Ar); // F2 real
            _mm256_store_pd(out_re, R4);
            ADD_DOUBLE_PTR(out_re, write_offset);
            SIMD_SUB(R5, Ci, Ai); // F2 im
            _mm256_store_pd(out_im, R5);
            ADD_DOUBLE_PTR(out_im, write_offset);

            SIMD_SUB(R6, Di, Br); // F3 real
            _mm256_store_pd(out_re, R6);
            SUB_DOUBLE_PTR(out_re, write_offset*3-4);
            SIMD_ADD(R7, Dr, Bi); // F3 im
            _mm256_store_pd(out_im, R7);
            SUB_DOUBLE_PTR(out_im, write_offset*3-4);
        }
        ADD_DOUBLE_PTR(in_re, 3*read_offset);
        ADD_DOUBLE_PTR(in_im, 3*read_offset);
        ADD_DOUBLE_PTR(out_re, 3*write_offset);
        ADD_DOUBLE_PTR(out_im, 3*write_offset);
    }
}

__inline__ void fft4_kernel_transpose(const double *in_re, const double *in_im, 
                 double *out_re, double *out_im, int n_doubles, int fft_size,
                 const double* twiddle_r, const double* twiddle_i, 
                 int read_offset, int write_offset, int stage) 
{
    for (int i = 0; i < n_doubles/fft_size; i++) {
        int iters = fft_size/16;
        for (int j = 0; j < iters; j++) {
            register __m256d R0 = _mm256_load_pd(in_re);   // X0 real
            ADD_DOUBLE_PTR(in_re, read_offset*2);
            register __m256d R2 = _mm256_load_pd(in_re);   // X2 real
            SUB_DOUBLE_PTR(in_re, read_offset);
            register __m256d R8;
            SIMD_ADD(R8, R0, R2);
            register __m256d R1 = _mm256_load_pd(in_re);   // X1 real
            ADD_DOUBLE_PTR(in_re, read_offset*2);
            register __m256d R3 = _mm256_load_pd(in_re);  // X3 real
            SUB_DOUBLE_PTR(in_re, read_offset*3-4);
            register __m256d R9;
            SIMD_ADD(R9, R1, R3);

            register __m256d R4 = _mm256_load_pd(in_im);   // X0 im
            ADD_DOUBLE_PTR(in_im, read_offset*2);
            register __m256d R6 = _mm256_load_pd(in_im);   // X2 im
            SUB_DOUBLE_PTR(in_im, read_offset);
            register __m256d R10;
            SIMD_ADD(R10, R4, R6);
            register __m256d R5 = _mm256_load_pd(in_im);   // X1 im
            ADD_DOUBLE_PTR(in_im, read_offset*2);
            register __m256d R7 = _mm256_load_pd(in_im);  // X3 im
            SUB_DOUBLE_PTR(in_im, read_offset*3-4);
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
            PROCESS_TWIDDLES_START(stage, R9, R10, twiddle_r, twiddle_i, read_offset, iters);
            SIMD_MUL(R8, R0, R10); //ad
            SIMD_MUL(R0, R0, R9); //ac
            SIMD_FMS(R0, R1, R10) //ac-bd F0 real
            SIMD_FMA(R8, R1, R9); //ad+bc F0 im

            PROCESS_TWIDDLES_START(stage, R10, R11, twiddle_r, twiddle_i, read_offset, iters);
            SIMD_MUL(R9, R2, R11); //ad
            SIMD_MUL(R1, R2, R10); //ac
            SIMD_FMS(R1, R3, R11); //ac-bd F1 real
            SIMD_FMA(R9, R3, R10); //ad+bc F1 im

            PROCESS_TWIDDLES_START(stage, R12, R13, twiddle_r, twiddle_i, read_offset, iters);
            SIMD_MUL(R10, R4, R13); //ad
            SIMD_MUL(R2, R4, R12); //ac
            SIMD_FMS(R2, R5, R13); //ac-bd F2 real
            R4 = _mm256_permute2f128_pd(R0, R2, 0x20); // [A0 B0 C0 D0]
            SIMD_FMA(R10, R5, R12); //ad+bc F2 im
            R5 = _mm256_permute2f128_pd(R0, R2, 0x31); // [A1 B1 C1 D1]

            PROCESS_TWIDDLES_END(stage, R13, R14, twiddle_r, twiddle_i, read_offset, iters);
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
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm256_store_pd(out_im, R8);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm256_store_pd(out_re, R1);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm256_store_pd(out_im, R9);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm256_store_pd(out_re, R2);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm256_store_pd(out_im, R10);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm256_store_pd(out_re, R3);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm256_store_pd(out_im, R11);
            ADD_DOUBLE_PTR(out_im, write_offset);
        }
        ADD_DOUBLE_PTR(in_re, 3*read_offset);
        ADD_DOUBLE_PTR(in_im, 3*read_offset);
    }
}

int main() {
    // 1. Setup Memory
    size_t n_doubles = 64;
    int fft_size = 64;
    size_t size = n_doubles * sizeof(double);
    
    double *in_re = (double*)_mm_malloc(size, 32);
    double *in_im = (double*)_mm_malloc(size, 32);
    double *mid_re = (double*)_mm_malloc(size, 32);
    double *mid_im = (double*)_mm_malloc(size, 32);
    double *out_re = (double*)_mm_malloc(size, 32);
    double *out_im = (double*)_mm_malloc(size, 32);

    // 2. Initialize Data
    for(int i=0; i<n_doubles; i++) { 
        in_re[i] = 1.0*i; 
        in_im[i] = 1.0*i;
    }
    double twiddle_r[fft_size], twiddle_i[fft_size];
    double pi = 3.14159265358979323846;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < fft_size/4; ++j) {
            double theta = -2.0 * pi * i * j / fft_size;
            twiddle_r[i*fft_size/4+j] = cos(theta);
            twiddle_i[i*fft_size/4+j] = sin(theta);
        }
    }

    int iters = log2(n_doubles)/2-1;
    unsigned long long start = rdtsc();

    for(int i=0; i<RUNS; i++) {
        double* i_re = in_re;
        double* i_im = in_im;
        double* o_re = mid_re;
        double* o_im = mid_im;
        double* temp;
        for(int j = 0; j < iters; j++) {
            fft4_kernel_transpose(i_re, i_im, o_re, o_im, 
                n_doubles, fft_size, twiddle_r, twiddle_i, fft_size/4, 4, j);
            temp = i_re;
            i_re = o_re;
            o_re = temp;
            temp = i_im;
            i_im = o_im;
            o_im = temp;
            if(j == 0) {o_re = out_re; o_im = out_im;}
        }
        fft4_kernel(i_re, i_im, out_re, out_im, n_doubles, fft_size, fft_size/4, fft_size/4);
    }
    
    unsigned long long end = rdtsc();

    unsigned long long total_cycles = end - start;
    double avg_cycles = ((double)total_cycles) / RUNS;
    double seconds = (double)total_cycles / (CPU_FREQ_GHZ * 1e9);
    
    double total_flops = (64.0 + 160.0*iters) * (fft_size/16.0) * n_doubles/fft_size * RUNS;
    double flopspercycle = total_flops / (double)total_cycles;

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
    for (int i = 0; i < n_doubles/fft_size; ++i) {
        for (int j = 0; j < sqrt(fft_size); ++j) {
            for (int k = 0; k < sqrt(fft_size); ++k) {
                printf("%.2f+%.2f\t", out_re[i*fft_size+j*4+k], out_im[i*fft_size+j*4+k]);		
            }
            printf("\n");
        }
        printf("\n");
    }

    _mm_free(in_re); _mm_free(in_im);
    _mm_free(out_re); _mm_free(out_im);

    return 0;
}
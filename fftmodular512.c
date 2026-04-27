#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>

#define RUNS 20000000
#define CPU_FREQ_GHZ 3.20

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
  : [rdest] "=v"(dest)      \
  : [rs1] "v"(s1), [rs2] "v"(s2)         \
);

#define SIMD_SUB(dest, s1, s2) \
__asm__ __volatile__ (      \
  "vsubpd %[rs2], %[rs1], %[rdest]\n" \
  : [rdest] "=v"(dest)      \
  : [rs1] "v"(s1), [rs2] "v"(s2)         \
);

#define SIMD_MUL(dest, s1, s2) \
__asm__ __volatile__ (      \
  "vmulpd %[rs1], %[rs2], %[rdest]\n" \
  : [rdest] "=v"(dest)      \
  : [rs1] "v"(s1), [rs2] "v"(s2)         \
);

#define SIMD_FMA(dest, a, s1, s2)                                  \
__asm__ __volatile__(                                              \
  "vfmadd231pd %[rs1], %[rs2], %[rd]\n\t" /* dest = a + s1*s2 */   \
  : [rd] "=v"(dest)                       /* Write-only output */  \
  : "0"(a), [rs1] "v"(s1), [rs2] "v"(s2)  /* Tie 'a' to [rd] */    \
);

#define SIMD_FMS(dest, a, s1, s2)                                  \
__asm__ __volatile__(                                              \
  "vfnmadd231pd %[rs1], %[rs2], %[rd]\n\t" /* dest = a - s1*s2 */  \
  : [rd] "=v"(dest)                        /* Write-only output */ \
  : "0"(a), [rs1] "v"(s1), [rs2] "v"(s2)   /* Tie 'a' to [rd] */   \
);


//fix this for avx512
#define PROCESS_TWIDDLES(stage, R1, R2, tr, ti, offset, iters) \
    if ((stage) == 0) { \
        (R1) = _mm512_load_pd(tr); \
        (tr) += (offset); \
        (R2) = _mm512_load_pd(ti); \
        (ti) += (offset); \
    } else { \
        __m512d bcast_tr0 = _mm512_set1_pd(*(tr)); \
        __m512d bcast_tr1 = _mm512_set1_pd(*(tr+1)); \
        (R1) = _mm512_mask_blend_pd(0x0F, bcast_tr0, bcast_tr1); \
        (tr) += (iters); \
        __m512d bcast_ti0 = _mm512_set1_pd(*(ti)); \
        __m512d bcast_ti1 = _mm512_set1_pd(*(ti+1)); \
        (R2) = _mm512_mask_blend_pd(0x0F, bcast_ti0, bcast_ti1); \
        (ti) += (iters); \
    } \


static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

__inline__ void fft8_kernel(double *in_re, double *in_im, 
                 double *out_re, double *out_im, int n_doubles, int fft_size, 
                 int read_offset, int write_offset) 
{   
    for (int i = 0; i < n_doubles/fft_size; i++) {
        int iters = fft_size/64;
        __m512d vC    = _mm512_set1_pd(0.70710678118654752440); // cos(-pi/4) and sin
        __m512d vNegC = _mm512_set1_pd(-0.70710678118654752440);
        __m512d vZero = _mm512_setzero_pd();
        for (int j = 0; j < iters; ++j) {
            __m512d R0 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R1 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R2 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R3 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R4 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R5 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R6 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R7 = _mm512_load_pd(in_re);
            SUB_DOUBLE_PTR(in_re, read_offset*7-8);

            __m512d I0 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I1 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I2 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I3 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I4 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I5 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I6 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I7 = _mm512_load_pd(in_im);
            SUB_DOUBLE_PTR(in_im, read_offset*7-8);

            //STAGE 1
            __m512d t0_r, t0_i, t1_r, t1_i;
            SIMD_ADD(t0_r, R0, R4); //intermediate 0
            SIMD_ADD(t0_i, I0, I4);
            SIMD_SUB(t1_r, R0, R4); //intermediate 1 
            SIMD_SUB(t1_i, I0, I4);

            __m512d t2_r, t2_i, t3_r, t3_i;
            SIMD_ADD(t2_r, R2, R6); //intermediate 2
            SIMD_ADD(t2_i, I2, I6);
            SIMD_SUB(t3_r, R2, R6); //intermediate 3
            SIMD_SUB(t3_i, I2, I6);

            __m512d t4_r, t4_i, t5_r, t5_i;
            SIMD_ADD(t4_r, R1, R5); //intermediate 4
            SIMD_ADD(t4_i, I1, I5);
            SIMD_SUB(t5_r, R1, R5); //intermediate 5
            SIMD_SUB(t5_i, I1, I5);

            __m512d t6_r, t6_i, t7_r, t7_i;
            SIMD_ADD(t6_r, R3, R7); //intermediate 6
            SIMD_ADD(t6_i, I3, I7);
            SIMD_SUB(t7_r, R3, R7); //intermediate 7
            SIMD_SUB(t7_i, I3, I7);

            // STAGE 2
            __m512d E0_r, E0_i, E1_r, E1_i, E2_r, E2_i, E3_r, E3_i;
            SIMD_ADD(E0_r, t0_r, t2_r); //intermediate 0
            SIMD_ADD(E0_i, t0_i, t2_i);
            SIMD_SUB(E2_r, t0_r, t2_r); //intermediate 2
            SIMD_SUB(E2_i, t0_i, t2_i);
            
            SIMD_ADD(E1_r, t1_r, t3_i); //intermediate 1
            SIMD_SUB(E1_i, t1_i, t3_r);
            SIMD_SUB(E3_r, t1_r, t3_i); //intermediate 3
            SIMD_ADD(E3_i, t1_i, t3_r);

            __m512d E4_r, E4_i, E5_r, E5_i, E6_r, E6_i, E7_r, E7_i;
            SIMD_ADD(E4_r, t4_r, t6_r); //intermediate 4
            SIMD_ADD(E4_i, t4_i, t6_i);
            SIMD_SUB(E6_r, t4_r, t6_r); //intermediate 5
            SIMD_SUB(E6_i, t4_i, t6_i);

            SIMD_ADD(E5_r, t5_r, t7_i); //intermediate 6
            SIMD_SUB(E5_i, t5_i, t7_r);
            SIMD_SUB(E7_r, t5_r, t7_i); //intermediate 7
            SIMD_ADD(E7_i, t5_i, t7_r);
            
            // STAGE 3
            __m512d X0_r, X0_i, X4_r, X4_i;
            SIMD_ADD(X0_r, E0_r, E4_r); //final 0
            SIMD_ADD(X0_i, E0_i, E4_i);
            SIMD_SUB(X4_r, E0_r, E4_r); //final 4
            SIMD_SUB(X4_i, E0_i, E4_i);

            __m512d X1_r, X1_i, X5_r, X5_i, ac, ad, ME5_r, ME5_i;
            SIMD_MUL(ac, E5_r, vC);
            SIMD_MUL(ad, E5_r, vNegC);
            SIMD_FMS(ME5_r, ac, E5_i, vNegC);
            SIMD_FMA(ME5_i, ad, E5_i, vC);
            SIMD_ADD(X1_r, E1_r, ME5_r); //final 1
            SIMD_SUB(X5_r, E1_r, ME5_r); //final 5
            SIMD_ADD(X1_i, E1_i, ME5_i);
            SIMD_SUB(X5_i, E1_i, ME5_i);

            __m512d X2_r, X2_i, X6_r, X6_i;
            SIMD_ADD(X2_r, E2_r, E6_i); //final 2
            SIMD_SUB(X2_i, E2_i, E6_r);
            SIMD_SUB(X6_r, E2_r, E6_i); //final 6
            SIMD_ADD(X6_i, E2_i, E6_r);

            __m512d X3_r, X3_i, X7_r, X7_i, ME7_r, ME7_i;
            SIMD_MUL(ac, E7_r, vNegC);
            SIMD_MUL(ad, E7_r, vNegC);
            SIMD_FMS(ME7_r, ac, E7_i, vNegC);
            SIMD_FMA(ME7_i, ad, E7_i, vNegC);
            SIMD_ADD(X3_r, E3_r, ME7_r); //final 3
            SIMD_ADD(X3_i, E3_i, ME7_i); //final 7
            SIMD_SUB(X7_r, E3_r, ME7_r); 
            SIMD_SUB(X7_i, E3_i, ME7_i);

            // Store results (Structure of Arrays)
            _mm512_store_pd(out_re, X0_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X0_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X1_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X1_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X2_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X2_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X3_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X3_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X4_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X4_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X5_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X5_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X6_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X6_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X7_r); 
            SUB_DOUBLE_PTR(out_re, write_offset*7-8);
            _mm512_store_pd(out_im, X7_i);
            SUB_DOUBLE_PTR(out_im, write_offset*7-8);
        }
        ADD_DOUBLE_PTR(in_re, 7*read_offset);
        ADD_DOUBLE_PTR(in_im, 7*read_offset);
        ADD_DOUBLE_PTR(out_re, 7*write_offset);
        ADD_DOUBLE_PTR(out_im, 7*write_offset);
    }
}

__inline__ void fft8_kernel_transpose(const double *in_re, const double *in_im, 
                 double *out_re, double *out_im, int n_doubles, int fft_size,
                 const double* twiddle_r, const double* twiddle_i, 
                 int read_offset, int write_offset, int stage) 
{
    double* twiddle_i_reset = twiddle_i;
    double* twiddle_r_reset = twiddle_r;
    for (int i = 0; i < n_doubles/fft_size; i++) {
        int iters = fft_size/64;
        __m512d vC    = _mm512_set1_pd(0.70710678118654752440);
        __m512d vNegC = _mm512_set1_pd(-0.70710678118654752440);
        __m512d vZero = _mm512_setzero_pd();
        for (int j = 0; j < iters; ++j) {
            __m512d R0 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R1 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R2 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R3 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R4 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R5 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R6 = _mm512_load_pd(in_re);
            ADD_DOUBLE_PTR(in_re, read_offset);
            __m512d R7 = _mm512_load_pd(in_re);
            SUB_DOUBLE_PTR(in_re, read_offset*7-8);

            __m512d I0 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I1 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I2 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I3 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I4 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I5 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I6 = _mm512_load_pd(in_im);
            ADD_DOUBLE_PTR(in_im, read_offset);
            __m512d I7 = _mm512_load_pd(in_im);
            SUB_DOUBLE_PTR(in_im, read_offset*7-8);

            //STAGE 1
            __m512d t0_r, t0_i, t1_r, t1_i;
            SIMD_ADD(t0_r, R0, R4); //intermediate 0
            SIMD_ADD(t0_i, I0, I4);
            SIMD_SUB(t1_r, R0, R4); //intermediate 1 
            SIMD_SUB(t1_i, I0, I4);

            __m512d t2_r, t2_i, t3_r, t3_i;
            SIMD_ADD(t2_r, R2, R6); //intermediate 2
            SIMD_ADD(t2_i, I2, I6);
            SIMD_SUB(t3_r, R2, R6); //intermediate 3
            SIMD_SUB(t3_i, I2, I6);

            __m512d t4_r, t4_i, t5_r, t5_i;
            SIMD_ADD(t4_r, R1, R5); //intermediate 4
            SIMD_ADD(t4_i, I1, I5);
            SIMD_SUB(t5_r, R1, R5); //intermediate 5
            SIMD_SUB(t5_i, I1, I5);

            __m512d t6_r, t6_i, t7_r, t7_i;
            SIMD_ADD(t6_r, R3, R7); //intermediate 6
            SIMD_ADD(t6_i, I3, I7);
            SIMD_SUB(t7_r, R3, R7); //intermediate 7
            SIMD_SUB(t7_i, I3, I7);

            // STAGE 2
            __m512d E0_r, E0_i, E1_r, E1_i, E2_r, E2_i, E3_r, E3_i;
            SIMD_ADD(E0_r, t0_r, t2_r); //intermediate 0
            SIMD_ADD(E0_i, t0_i, t2_i);
            SIMD_SUB(E2_r, t0_r, t2_r); //intermediate 2
            SIMD_SUB(E2_i, t0_i, t2_i);
            
            SIMD_ADD(E1_r, t1_r, t3_i); //intermediate 1
            SIMD_SUB(E1_i, t1_i, t3_r);
            SIMD_SUB(E3_r, t1_r, t3_i); //intermediate 3
            SIMD_ADD(E3_i, t1_i, t3_r);

            __m512d E4_r, E4_i, E5_r, E5_i, E6_r, E6_i, E7_r, E7_i;
            SIMD_ADD(E4_r, t4_r, t6_r); //intermediate 4
            SIMD_ADD(E4_i, t4_i, t6_i);
            SIMD_SUB(E6_r, t4_r, t6_r); //intermediate 5
            SIMD_SUB(E6_i, t4_i, t6_i);

            SIMD_ADD(E5_r, t5_r, t7_i); //intermediate 6
            SIMD_SUB(E5_i, t5_i, t7_r);
            SIMD_SUB(E7_r, t5_r, t7_i); //intermediate 7
            SIMD_ADD(E7_i, t5_i, t7_r);
            
            // STAGE 3
            __m512d X0_r, X0_i, X4_r, X4_i;
            SIMD_ADD(X0_r, E0_r, E4_r); //final 0
            SIMD_ADD(X0_i, E0_i, E4_i);
            SIMD_SUB(X4_r, E0_r, E4_r); //final 4
            SIMD_SUB(X4_i, E0_i, E4_i);

            __m512d X1_r, X1_i, X5_r, X5_i, ac, ad, ME5_r, ME5_i;
            SIMD_MUL(ac, E5_r, vC);
            SIMD_MUL(ad, E5_r, vNegC);
            SIMD_FMS(ME5_r, ac, E5_i, vNegC);
            SIMD_FMA(ME5_i, ad, E5_i, vC);
            SIMD_ADD(X1_r, E1_r, ME5_r); //final 1
            SIMD_SUB(X5_r, E1_r, ME5_r); //final 5
            SIMD_ADD(X1_i, E1_i, ME5_i);
            SIMD_SUB(X5_i, E1_i, ME5_i);

            __m512d X2_r, X2_i, X6_r, X6_i;
            SIMD_ADD(X2_r, E2_r, E6_i); //final 2
            SIMD_SUB(X2_i, E2_i, E6_r);
            SIMD_SUB(X6_r, E2_r, E6_i); //final 6
            SIMD_ADD(X6_i, E2_i, E6_r);

            __m512d X3_r, X3_i, X7_r, X7_i, ME7_r, ME7_i;
            SIMD_MUL(ac, E7_r, vNegC);
            SIMD_MUL(ad, E7_r, vNegC);
            SIMD_FMS(ME7_r, ac, E7_i, vNegC);
            SIMD_FMA(ME7_i, ad, E7_i, vNegC);
            SIMD_ADD(X3_r, E3_r, ME7_r); //final 3
            SIMD_ADD(X3_i, E3_i, ME7_i); //final 7
            SIMD_SUB(X7_r, E3_r, ME7_r); 
            SIMD_SUB(X7_i, E3_i, ME7_i);

            __m512d W0_r = _mm512_load_pd(twiddle_r); ADD_DOUBLE_PTR(twiddle_r, read_offset);
            __m512d W1_r = _mm512_load_pd(twiddle_r); ADD_DOUBLE_PTR(twiddle_r, read_offset);
            __m512d W2_r = _mm512_load_pd(twiddle_r); ADD_DOUBLE_PTR(twiddle_r, read_offset);
            __m512d W3_r = _mm512_load_pd(twiddle_r); ADD_DOUBLE_PTR(twiddle_r, read_offset);
            __m512d W4_r = _mm512_load_pd(twiddle_r); ADD_DOUBLE_PTR(twiddle_r, read_offset);
            __m512d W5_r = _mm512_load_pd(twiddle_r); ADD_DOUBLE_PTR(twiddle_r, read_offset);
            __m512d W6_r = _mm512_load_pd(twiddle_r); ADD_DOUBLE_PTR(twiddle_r, read_offset);
            __m512d W7_r = _mm512_load_pd(twiddle_r); 
            SUB_DOUBLE_PTR(twiddle_r, read_offset * 7 - 8);

            __m512d W0_i = _mm512_load_pd(twiddle_i); ADD_DOUBLE_PTR(twiddle_i, read_offset);
            __m512d W1_i = _mm512_load_pd(twiddle_i); ADD_DOUBLE_PTR(twiddle_i, read_offset);
            __m512d W2_i = _mm512_load_pd(twiddle_i); ADD_DOUBLE_PTR(twiddle_i, read_offset);
            __m512d W3_i = _mm512_load_pd(twiddle_i); ADD_DOUBLE_PTR(twiddle_i, read_offset);
            __m512d W4_i = _mm512_load_pd(twiddle_i); ADD_DOUBLE_PTR(twiddle_i, read_offset);
            __m512d W5_i = _mm512_load_pd(twiddle_i); ADD_DOUBLE_PTR(twiddle_i, read_offset);
            __m512d W6_i = _mm512_load_pd(twiddle_i); ADD_DOUBLE_PTR(twiddle_i, read_offset);
            __m512d W7_i = _mm512_load_pd(twiddle_i); 
            SUB_DOUBLE_PTR(twiddle_i, read_offset * 7 - 8);

            // Temporary products for R*Wr and R*Wi
            __m512d tr, ti;

            // X0: (R*Wr - I*Wi) + i(R*Wi + I*Wr)
            SIMD_MUL(tr, X0_r, W0_r);
            SIMD_MUL(ti, X0_r, W0_i);
            SIMD_FMS(X0_r, tr, X0_i, W0_i);
            SIMD_FMA(X0_i, ti, X0_i, W0_r);

            // X1
            SIMD_MUL(tr, X1_r, W1_r);
            SIMD_MUL(ti, X1_r, W1_i);
            SIMD_FMS(X1_r, tr, X1_i, W1_i);
            SIMD_FMA(X1_i, ti, X1_i, W1_r);

            // X2
            SIMD_MUL(tr, X2_r, W2_r);
            SIMD_MUL(ti, X2_r, W2_i);
            SIMD_FMS(X2_r, tr, X2_i, W2_i);
            SIMD_FMA(X2_i, ti, X2_i, W2_r);

            // X3
            SIMD_MUL(tr, X3_r, W3_r);
            SIMD_MUL(ti, X3_r, W3_i);
            SIMD_FMS(X3_r, tr, X3_i, W3_i);
            SIMD_FMA(X3_i, ti, X3_i, W3_r);

            // X4
            SIMD_MUL(tr, X4_r, W4_r);
            SIMD_MUL(ti, X4_r, W4_i);
            SIMD_FMS(X4_r, tr, X4_i, W4_i);
            SIMD_FMA(X4_i, ti, X4_i, W4_r);

            // X5
            SIMD_MUL(tr, X5_r, W5_r);
            SIMD_MUL(ti, X5_r, W5_i);
            SIMD_FMS(X5_r, tr, X5_i, W5_i);
            SIMD_FMA(X5_i, ti, X5_i, W5_r);

            // X6
            SIMD_MUL(tr, X6_r, W6_r);
            SIMD_MUL(ti, X6_r, W6_i);
            SIMD_FMS(X6_r, tr, X6_i, W6_i);
            SIMD_FMA(X6_i, ti, X6_i, W6_r);

            // X7
            SIMD_MUL(tr, X7_r, W7_r);
            SIMD_MUL(ti, X7_r, W7_i);
            SIMD_FMS(X7_r, tr, X7_i, W7_i);
            SIMD_FMA(X7_i, ti, X7_i, W7_r);

            __m512d r_t0 = _mm512_shuffle_pd(X0_r, X1_r, 0x00);
            __m512d r_t1 = _mm512_shuffle_pd(X0_r, X1_r, 0xFF);
            __m512d r_t2 = _mm512_shuffle_pd(X2_r, X3_r, 0x00);
            __m512d r_t3 = _mm512_shuffle_pd(X2_r, X3_r, 0xFF);
            __m512d r_t4 = _mm512_shuffle_pd(X4_r, X5_r, 0x00);
            __m512d r_t5 = _mm512_shuffle_pd(X4_r, X5_r, 0xFF);
            __m512d r_t6 = _mm512_shuffle_pd(X6_r, X7_r, 0x00);
            __m512d r_t7 = _mm512_shuffle_pd(X6_r, X7_r, 0xFF);

            // 128-bit interleave
            __m512d r_s0 = _mm512_shuffle_f64x2(r_t0, r_t2, 0x44);
            __m512d r_s1 = _mm512_shuffle_f64x2(r_t1, r_t3, 0x44);
            __m512d r_s2 = _mm512_shuffle_f64x2(r_t0, r_t2, 0xEE);
            __m512d r_s3 = _mm512_shuffle_f64x2(r_t1, r_t3, 0xEE);
            __m512d r_s4 = _mm512_shuffle_f64x2(r_t4, r_t6, 0x44);
            __m512d r_s5 = _mm512_shuffle_f64x2(r_t5, r_t7, 0x44);
            __m512d r_s6 = _mm512_shuffle_f64x2(r_t4, r_t6, 0xEE);
            __m512d r_s7 = _mm512_shuffle_f64x2(r_t5, r_t7, 0xEE);

            // 256-bit interleave
            X0_r = _mm512_shuffle_f64x2(r_s0, r_s4, 0x88);
            X1_r = _mm512_shuffle_f64x2(r_s1, r_s5, 0x88);
            X2_r = _mm512_shuffle_f64x2(r_s0, r_s4, 0xDD);
            X3_r = _mm512_shuffle_f64x2(r_s1, r_s5, 0xDD);
            X4_r = _mm512_shuffle_f64x2(r_s2, r_s6, 0x88);
            X5_r = _mm512_shuffle_f64x2(r_s3, r_s7, 0x88);
            X6_r = _mm512_shuffle_f64x2(r_s2, r_s6, 0xDD);
            X7_r = _mm512_shuffle_f64x2(r_s3, r_s7, 0xDD);

            // Transpose Imaginaries
            // 64-bit interleave
            __m512d i_t0 = _mm512_shuffle_pd(X0_i, X1_i, 0x00);
            __m512d i_t1 = _mm512_shuffle_pd(X0_i, X1_i, 0xFF);
            __m512d i_t2 = _mm512_shuffle_pd(X2_i, X3_i, 0x00);
            __m512d i_t3 = _mm512_shuffle_pd(X2_i, X3_i, 0xFF);
            __m512d i_t4 = _mm512_shuffle_pd(X4_i, X5_i, 0x00);
            __m512d i_t5 = _mm512_shuffle_pd(X4_i, X5_i, 0xFF);
            __m512d i_t6 = _mm512_shuffle_pd(X6_i, X7_i, 0x00);
            __m512d i_t7 = _mm512_shuffle_pd(X6_i, X7_i, 0xFF);

            //128-bit interleave
            __m512d i_s0 = _mm512_shuffle_f64x2(i_t0, i_t2, 0x44);
            __m512d i_s1 = _mm512_shuffle_f64x2(i_t1, i_t3, 0x44);
            __m512d i_s2 = _mm512_shuffle_f64x2(i_t0, i_t2, 0xEE);
            __m512d i_s3 = _mm512_shuffle_f64x2(i_t1, i_t3, 0xEE);
            __m512d i_s4 = _mm512_shuffle_f64x2(i_t4, i_t6, 0x44);
            __m512d i_s5 = _mm512_shuffle_f64x2(i_t5, i_t7, 0x44);
            __m512d i_s6 = _mm512_shuffle_f64x2(i_t4, i_t6, 0xEE);
            __m512d i_s7 = _mm512_shuffle_f64x2(i_t5, i_t7, 0xEE);

            //256-bit interleave
            X0_i = _mm512_shuffle_f64x2(i_s0, i_s4, 0x88);
            X1_i = _mm512_shuffle_f64x2(i_s1, i_s5, 0x88);
            X2_i = _mm512_shuffle_f64x2(i_s0, i_s4, 0xDD);
            X3_i = _mm512_shuffle_f64x2(i_s1, i_s5, 0xDD);
            X4_i = _mm512_shuffle_f64x2(i_s2, i_s6, 0x88);
            X5_i = _mm512_shuffle_f64x2(i_s3, i_s7, 0x88);
            X6_i = _mm512_shuffle_f64x2(i_s2, i_s6, 0xDD);
            X7_i = _mm512_shuffle_f64x2(i_s3, i_s7, 0xDD);

            // Store results 
            _mm512_store_pd(out_re, X0_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X0_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X1_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X1_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X2_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X2_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X3_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X3_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X4_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X4_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X5_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X5_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X6_r);
            ADD_DOUBLE_PTR(out_re, write_offset);
            _mm512_store_pd(out_im, X6_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
            _mm512_store_pd(out_re, X7_r);
            ADD_DOUBLE_PTR(out_re, write_offset); 
            _mm512_store_pd(out_im, X7_i);
            ADD_DOUBLE_PTR(out_im, write_offset);
        }
        ADD_DOUBLE_PTR(in_re, 7*read_offset);
        ADD_DOUBLE_PTR(in_im, 7*read_offset);
        twiddle_r = twiddle_r_reset;
        twiddle_i = twiddle_i_reset;
    }
}

int main() {
    // 1. Setup Memory
    size_t n_doubles = 256;
    int fft_size = 64;
    size_t size = n_doubles * sizeof(double);
    
    double *in_re = (double*)_mm_malloc(size, 64);
    double *in_im = (double*)_mm_malloc(size, 64);
    double *mid_re = (double*)_mm_malloc(size, 64);
    double *mid_im = (double*)_mm_malloc(size, 64);
    double *out_re = (double*)_mm_malloc(size, 64);
    double *out_im = (double*)_mm_malloc(size, 64);
    // 2. Initialize Data
    for(int i=0; i<n_doubles/fft_size; i++) {
        for(int j = 0; j < fft_size; j++) {
            in_re[i*fft_size+j] = 1.0*j; 
            in_im[i*fft_size+j] = 1.0*j;
        }
    }
    
    double twiddle_r[fft_size], twiddle_i[fft_size];
    double pi = 3.14159265358979323846;
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < fft_size/8; ++j) {
            double theta = -2.0 * pi * i * j / fft_size;
            twiddle_r[i*fft_size/8+j] = cos(theta);
            twiddle_i[i*fft_size/8+j] = sin(theta);
        }
    }

    int iters = log2(fft_size)/3-1;
    unsigned long long start = rdtsc();

    for(int i=0; i<RUNS; i++) {
        double* i_re = in_re;
        double* i_im = in_im;
        double* o_re = mid_re;
        double* o_im = mid_im;
        double* temp;
        for(int j = 0; j < iters; j++) {
            fft8_kernel_transpose(i_re, i_im, o_re, o_im, 
                n_doubles, fft_size, twiddle_r, twiddle_i, fft_size/8, 8, j);
            temp = i_re;
            i_re = o_re;
            o_re = temp;
            temp = i_im;
            i_im = o_im;
            o_im = temp;
            if(j == 0) {o_re = out_re; o_im = out_im;}
        }
        fft8_kernel(i_re, i_im, out_re, out_im, n_doubles, fft_size, fft_size/8, fft_size/8);
    }
    
    unsigned long long end = rdtsc();

    unsigned long long total_cycles = end - start;
    double avg_cycles = ((double)total_cycles) / RUNS;
    double seconds = (double)total_cycles / (CPU_FREQ_GHZ * 1e9);
    
    double total_flops = (448.0 + 832*iters) * (fft_size/64.0) * n_doubles/fft_size * RUNS;
    double flopspercycle = total_flops / (double)total_cycles;

    printf("--- Performance Report ---\n");
    printf("CPU Frequency:  %.2f GHz\n", CPU_FREQ_GHZ);
    printf("Total Runs:     %d\n", RUNS);
    printf("Total Cycles:   %llu\n", total_cycles);
    printf("Avg Cycles/Run: %.2f\n", avg_cycles);
    printf("Est. Time (s):  %.6f\n", seconds);
    printf("Throughput:     %.2f FLOPS PER CYCLE\n", flopspercycle);
    printf("--------------------------\n");

    for (int i = 0; i < n_doubles/fft_size; ++i) {
        int len = sqrt(fft_size);
        for (int j = 0; j < len; ++j) {
            for (int k = 0; k < len; ++k) {
                printf("%.2f %.2f\t", out_re[i*fft_size+j*len+k], out_im[i*fft_size+j*len+k]);		
            }
            printf("\n");
        }
        printf("\n");
    }

    _mm_free(in_re); _mm_free(in_im);
    _mm_free(out_re); _mm_free(out_im);

    return 0;
}
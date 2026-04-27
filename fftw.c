#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>

#define RUNS 200000000
#define BATCHES 32
#define CPU_FREQ_GHZ 3.20

static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

void print_performance(const char* format_name, unsigned long long start, unsigned long long end, int loops) {
    unsigned long long total_cycles = end - start;
    double avg_cycles_per_batch = ((double)total_cycles) / loops;
    double avg_cycles_per_fft = avg_cycles_per_batch / BATCHES;

    printf("--- Performance Report: %s ---\n", format_name);
    printf("CPU Frequency:    %.2f GHz\n", CPU_FREQ_GHZ);
    printf("Batch Size:       %d\n", BATCHES);
    printf("Batch Runs:       %d\n", loops);
    printf("Total FFTs:       %d\n", loops * BATCHES);
    printf("Total Cycles:     %llu\n", total_cycles);
    printf("Avg Cycles/Batch: %.2f\n", avg_cycles_per_batch);
    printf("Avg Cycles/FFT:   %.2f\n\n", avg_cycles_per_fft);
}

void fft_interleaved_batch(int N, int batches) {
    fftw_complex *in, *out;
    fftw_plan p;
    
    int total_elements = N * batches;

    // Advanced interface parameters
    int n[] = {N};
    int idist = N, odist = N;   // Distance between the start of each FFT
    int istride = 1, ostride = 1; // Elements within an FFT are contiguous

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_elements);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * total_elements);
    
    p = fftw_plan_many_dft(
        1, n, batches,
        in, NULL, istride, idist,
        out, NULL, ostride, odist,
        FFTW_FORWARD, FFTW_MEASURE
    );

    // Initialize all batches
    for(int i = 0; i < total_elements; i++) {
        in[i][0] = 1.0 * (i % N); 
        in[i][1] = 1.0 * (i % N);
    }

    unsigned long long start = rdtsc();
    for (int i = 0; i < RUNS; i++) {
        fftw_execute(p);
    }
    unsigned long long end = rdtsc();

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    print_performance("Interleaved (Many DFT)", start, end, RUNS);
}

void fft_split_batch(int N, int batches) {
    int total_elements = N * batches;
    
    double *real_in  = (double*) fftw_malloc(sizeof(double) * total_elements);
    double *imag_in  = (double*) fftw_malloc(sizeof(double) * total_elements);
    double *real_out = (double*) fftw_malloc(sizeof(double) * total_elements);
    double *imag_out = (double*) fftw_malloc(sizeof(double) * total_elements);

    // Dimension of a single FFT
    fftw_iodim dim;
    dim.n = N;
    dim.is = 1;
    dim.os = 1;
    
    // Dimension defining the batching
    fftw_iodim howmany_dim;
    howmany_dim.n = batches;
    howmany_dim.is = N;
    howmany_dim.os = N;

    fftw_plan p = fftw_plan_guru_split_dft(
        1, &dim,
        1, &howmany_dim,
        real_in, imag_in,
        real_out, imag_out,
        FFTW_MEASURE
    );

    // Initialize all batches
    for(int i = 0; i < total_elements; i++) {
        real_in[i] = 1.0 * (i % N); 
        imag_in[i] = 1.0 * (i % N);
    }

    unsigned long long start = rdtsc();
    for (int i = 0; i < RUNS; i++) {
        fftw_execute_split_dft(p, real_in, imag_in, real_out, imag_out);
    }
    unsigned long long end = rdtsc();
    
    fftw_destroy_plan(p);
    fftw_free(real_in);
    fftw_free(imag_in);
    fftw_free(real_out);
    fftw_free(imag_out);

    print_performance("Split (Guru DFT)", start, end, RUNS);
}

int main() {
    // Computing total workload of TOTAL_FFTS, batched together
    fft_interleaved_batch(8, BATCHES);
    fft_split_batch(8, BATCHES);
    
    return 0;
}
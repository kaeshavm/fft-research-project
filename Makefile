CC = gcc
CFLAGS = -mavx -mavx2 -O3 -std=c99 -march=native -mfma
LDLIBS = -lm -lfftw3

# Define the targets
TARGETS = fftkernel.x fftkernel16.x fftmodular.x fftw.x fftmodular512.x fftkernel512.x

all: $(TARGETS)

# Rule for the original fft.x
fftkernel.x: fftkernel.c
	$(CC) $(CFLAGS) $(OBJS) fftkernel.c -o fftkernel.x $(LDLIBS)

# Rule for the new fft16.x
fftkernel16.x: fftkernel16.c
	$(CC) $(CFLAGS) $(OBJS) fftkernel16.c -o fftkernel16.x $(LDLIBS)

fftmodular.x: fftmodular.c
	$(CC) $(CFLAGS) $(OBJS) fftmodular.c -o fftmodular.x $(LDLIBS)

fftmodular512.x: fftmodular512.c
	$(CC) $(CFLAGS) -mavx512f $(OBJS) fftmodular512.c -o fftmodular512.x $(LDLIBS) 

fftkernel512.x: fftkernel512.c
	$(CC) $(CFLAGS) -mavx512f $(OBJS) fftkernel512.c -o fftkernel512.x $(LDLIBS) 

fftw.x: fftw.c
	$(CC) $(CFLAGS) $(OBJS) fftw.c -o fftw.x $(LDLIBS)

# Updated run command to execute both, or you can specify one
run:
	./fft.x
	./fft16.x
	./fftmodular.x
	./fftmodular512.x
clean:
	rm -f *.x *~ *.o
CC = gcc
CFLAGS = -mavx -mavx2 -O3 -std=c99 -march=native -mfma
LDLIBS = -lm

# Define the targets
TARGETS = fft.x fft16.x fftmodular.x

all: $(TARGETS)

# Rule for the original fft.x
fft.x: fftkernel.c
	$(CC) $(CFLAGS) $(OBJS) fftkernel.c -o fft.x $(LDLIBS)

# Rule for the new fft16.x
fft16.x: fftkernel16.c
	$(CC) $(CFLAGS) $(OBJS) fftkernel16.c -o fft16.x $(LDLIBS)

fftmodular.x: fftmodular.c
	$(CC) $(CFLAGS) $(OBJS) fftmodular.c -o fftmodular.x $(LDLIBS)

# Updated run command to execute both, or you can specify one
run:
	./fft.x
	./fft16.x
	./fftmodular.x

clean:
	rm -f *.x *~ *.o
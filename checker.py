import numpy as np
import argparse
import sys

def main():
    # Set up argument parser to take N from the command line
    parser = argparse.ArgumentParser(description="Calculate the FFT of complex numbers (i + i*j) from i=0 to N.")
    parser.add_argument("N", type=int, help="The maximum index (integer) in the sequence 0 to N.")
    parser.add_argument("-o", "--output", type=str, default="fft_output.txt", 
                        help="Output file name (default: fft_output.txt)")
    
    args = parser.parse_args()
    
    if args.N < 0:
        print("Error: N must be a non-negative integer.")
        sys.exit(1)

    # Generate the sequence where real = i and imaginary = i
    indices = np.arange(args.N + 1)
    numbers = indices + 1j * indices
    
    # Calculate the FFT
    fft_result = np.fft.fft(numbers)
    
    # Write the output to the specified file
    try:
        with open(args.output, 'w') as f:
            for i, val in enumerate(fft_result):
                # Formatting the complex number to handle the +/- signs cleanly
                f.write(f"{val.real:.2f} {val.imag:.2f}j\n")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

if __name__ == "__main__":
    main()
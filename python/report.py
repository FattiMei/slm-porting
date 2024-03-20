import numpy as np
import matplotlib.pyplot as plt
import sys


def read_binary_file(filename):
    SIZEOF_FLOAT64 = np.dtype(np.float64).itemsize

    with open(filename, 'rb') as f:
        width, height = [int(x) for x in f.readline().split()]
        buffer = f.read(SIZEOF_FLOAT64 * width * height)

        return np.frombuffer(buffer, dtype = np.float64).reshape(width, height)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python report.py <output_binary_file>")
        sys.exit(1)

    phase = read_binary_file(sys.argv[1])


    plt.imshow(phase, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Phase visualization')
    plt.show()

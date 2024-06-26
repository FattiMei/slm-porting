import numpy as np
import refactor
import matplotlib.pyplot as plt
import sys


def read_binary_vector(f):
    SIZEOF_FLOAT64 = np.dtype(np.float64).itemsize

    width, height = [int(x) for x in f.readline().split()]
    buffer = f.read(SIZEOF_FLOAT64 * width * height)

    return np.frombuffer(buffer, dtype = np.float64).reshape(width, height)


def read_binary_file(filename):
    with open(filename, 'rb') as f:
        out = read_binary_vector(f)
        pists = read_binary_vector(f)
        spots = read_binary_vector(f)

        return out, pists, spots


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python report.py <cpp_binary_file>")
        sys.exit(1)


    alternative, PISTS, SPOTS = read_binary_file(sys.argv[1])
    PISTS = PISTS.reshape(-1)


    FOCAL_LENGTH = 20.0
    PIXELS       = 512
    PITCH        = 15.0
    WAVELENGTH   = 0.488
    NPOINTS      = 100
    ITERATIONS   = 30
    COMPRESSION  = 0.05


    # reference, _ = refactor.rs(SPOTS,PISTS,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS)
    reference, _ = refactor.gs(SPOTS,PISTS,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS)
    # reference, _ = refactor.wgs(SPOTS,PISTS,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS)
    # reference, _ = refactor.csgs(SPOTS,PISTS,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,1)
    # reference, _ = refactor.wcsgs(SPOTS,PISTS,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,1)

    abs_err = np.max(np.abs(reference - alternative))
    nnz = np.where(reference != 0.0)
    rel_err = np.max(np.abs((reference[nnz] - alternative[nnz]) / reference[nnz]))

    print(f"Testing cpp and python, absolute error {abs_err}, relative error {rel_err}")

    try:
        fig, axis = plt.subplots(1,2)

        axis[0].imshow(reference, cmap='viridis', interpolation='nearest')
        axis[0].set_title('Reference')

        axis[1].imshow(alternative, cmap='viridis', interpolation='nearest')
        axis[1].set_title('C++')

        plt.show()


        plt.plot(np.abs(reference - alternative).flatten())
        plt.show()
    except:
        pass

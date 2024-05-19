import numpy as np
import matplotlib.pyplot as plt


RESOLUTION      = 512
OMP_NUM_THREADS = 4


def reverse_search(ordered, x):
    # lame search
    for i in range(len(ordered)):
        if x < ordered[i]:
            break

    return i


if __name__ == '__main__':
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, RESOLUTION),
        np.linspace(-1, 1, RESOLUTION)
    )

    points_in_line = np.sum((xx ** 2 + yy ** 2) < 1.0, axis=0)
    cumulative     = np.cumsum(points_in_line)
    cumulative     = cumulative / cumulative[-1]


    bounds = [reverse_search(cumulative, alpha) for alpha in np.linspace(0, 1, OMP_NUM_THREADS + 1)]
    print(bounds)


    plt.plot(cumulative)
    plt.show()

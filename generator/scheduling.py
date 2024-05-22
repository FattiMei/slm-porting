import numpy as np
import os
import config


# lame search but it works
def reverse_search(ordered, x):
    for i in range(len(ordered) - 1):
        if x >= ordered[i] and x < ordered[i+1]:
            return i

    return len(ordered)


if __name__ == '__main__':
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, config.RESOLUTION),
        np.linspace(-1, 1, config.RESOLUTION)
    )

    points_in_line = np.sum((xx ** 2 + yy ** 2) < 1.0, axis=1)
    cumulative     = np.cumsum(points_in_line, dtype = np.float64)

    bounds = [reverse_search(cumulative, alpha) for alpha in np.linspace(cumulative[0], cumulative[-1], config.OMP_NUM_THREADS + 1)]

    print(f'extern const int for_loop_bounds[{len(bounds)}] = {{')

    for x in bounds:
        print(f'\t{x},')

    print('};')

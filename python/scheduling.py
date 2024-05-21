import numpy as np


RESOLUTION      = 512
OMP_NUM_THREADS = 4


# lame search but it works
def reverse_search(ordered, x):
    for i in range(len(ordered) - 1):
        if x >= ordered[i] and x < ordered[i+1]:
            return i

    return len(ordered)


def format_bounds(bounds):
    print('const int for_loop_bounds[] = {')

    for x in bounds:
        print(f'\t{x},')

    print('};')


if __name__ == '__main__':
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, RESOLUTION),
        np.linspace(-1, 1, RESOLUTION)
    )

    points_in_line = np.sum((xx ** 2 + yy ** 2) < 1.0, axis=1)
    cumulative     = np.cumsum(points_in_line, dtype = np.float64)

    print(cumulative)
    print(np.linspace(cumulative[0], cumulative[-1], OMP_NUM_THREADS + 1))

    bounds = [reverse_search(cumulative, alpha) for alpha in np.linspace(cumulative[0], cumulative[-1], OMP_NUM_THREADS + 1)]
    format_bounds(bounds)

import numpy as np
import sys


WIDTH  = 512
HEIGHT = 512


def format_indices(indices):
    if len(indices):
        print(f'extern const int pupil_count = {len(indices)};')
        print(f'extern const int pupil_indices[] = {{')

        for i in indices:
            print(f'\t{i},')

        print('};')


if __name__ == '__main__':
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, WIDTH),
        np.linspace(-1, 1, HEIGHT)
    )

    indices = []
    count = 0

    for is_pupil in (xx ** 2 + yy ** 2 < 1.0).flatten():
        if is_pupil:
            indices.append(count)

        count += 1

    format_indices(indices)

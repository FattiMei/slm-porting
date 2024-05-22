import numpy as np
import sys
import os
import config


if __name__ == '__main__':
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, config.RESOLUTION),
        np.linspace(-1, 1, config.RESOLUTION)
    )
    mesh  = (xx ** 2 + yy ** 2 < 1.0)


    print('#include <utility>\n')
    print(f'extern const std::pair<int, int> pupil_index_bounds[{config.RESOLUTION}] = {{')

    for row in mesh:
        nnz = np.flatnonzero(row)

        if len(nnz):
            lower = nnz[0]
            upper = nnz[-1]
        else:
            lower = 0
            upper = 0

        print(f'\t{{{lower}, {upper}}},')

    print('};\n')


    indices = np.flatnonzero(mesh)

    print(f'extern const int pupil_count = {len(indices)};')
    print(f'extern const int pupil_indices[{len(indices)}] = {{')

    for i in indices:
        print(f'\t{i},')

    print('};')

import units
from units import Length
import backend_numpy
from slm import SLM
from rng import NumpyRng

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


if __name__ == '__main__':
    slm = SLM(
        focal      = units.Length(20.0, units.MILLI),
        pixel_size = units.Length(15.0, units.MICRO),
        wavelength = units.Length(488.0, units.NANO),
        resolution = 512,
        dtype = np.float64
    )
    NPOINTS = 100

    x = 100.0 * (np.random.random(NPOINTS)-0.5)
    y = 100.0 * (np.random.random(NPOINTS)-0.5)
    z =  10.0 * (np.random.random(NPOINTS)-0.5)

    start_time = perf_counter()
    phase = backend_numpy.rs_soa_pupil(
        x, y, z,
        slm.xx, slm.yy,
        slm.C1, slm.C2,
        NumpyRng(seed=42)
    )
    end_time = perf_counter()
    delta = end_time - start_time

    metrics = backend_numpy.compute_metrics_soa_pupil(
        x, y, z,
        slm.xx, slm.yy,
        slm.C1, slm.C2,
        phase
    )

    print(f'rs algorithm (NPOINTS={NPOINTS}): {delta:.2g} s')
    print(metrics)

    plt.title("rs algorithm")
    surf = np.zeros((slm.resolution, slm.resolution))
    surf[slm.pupil_idx] = phase
    plt.imshow(surf)
    plt.show()

import units
from units import Length
from slm import SLM
from rng import NumpyRng
from algorithms import rs, compute_quality_metrics

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
    phase = rs(slm, x, y, z, NumpyRng(seed=42))
    end_time = perf_counter()
    delta = end_time - start_time

    metrics = compute_quality_metrics(slm, x, y, z, phase)

    print(f'rs algorithm (NPOINTS={NPOINTS}): {delta:.2g} s')
    print(metrics)

    plt.title("rs algorithm")
    plt.imshow(phase)
    plt.show()

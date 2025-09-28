import dependency_manager
np = dependency_manager.dep('numpy')

from slm import SLM
from time import perf_counter
import matplotlib.pyplot as plt


if __name__ == '__main__':
    slm = SLM.get_standard_slm()
    backends = dependency_manager.get_available_backends()
    executor = backends['numpy'].get_executor(slm)

    NPOINTS = 100
    x = 100.0 * (np.random.random(NPOINTS)-0.5)
    y = 100.0 * (np.random.random(NPOINTS)-0.5)
    z =  10.0 * (np.random.random(NPOINTS)-0.5)
    pists = np.random.random(NPOINTS)

    tstart = perf_counter()
    phase = np.zeros((slm.resolution, slm.resolution))
    phase[slm.pupil_idx] = executor.rs(x,y,z,pists)
    delta = perf_counter() - tstart

    print(f'rs algorithm (NPOINTS={NPOINTS}): {delta:.2g} s')
    plt.title("rs algorithm")
    plt.imshow(phase)
    plt.show()

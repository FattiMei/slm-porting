import argparse
import matplotlib.pyplot as plt

from time import perf_counter
from slm.common.slm import SLM
from slm.common.loader import load, get_available_backends, print_available_backends
np = load('numpy')


def parse_command_line():
    parser = argparse.ArgumentParser(
        prog='visualization',
        description='Try a backend implementation of the `rs` algorithm'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='seed to initialize the RNG',
        default=42
    )

    parser.add_argument(
        '--backend',
        type=str,
        help='Select the implementation (cpu, gpu, different ones)',
        default='numpy'
    )

    parser.add_argument(
        '--nspots',
        type=int,
        help='The number of points to be generated',
        default=100
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line()
    seed = args.seed
    requested_backend = args.backend
    nspots = args.nspots

    slm = SLM.get_standard_slm()
    available_backends = get_available_backends()

    try:
        executor = available_backends[requested_backend].get_executor(slm)
    except KeyError:
        print(f'Requested backend "{requested_backend}" is not supported')
        print_available_backends(available_backends)
        exit()

    rng = np.random.default_rng(seed)
    x = 100.0 * (rng.random(nspots)-0.5)
    y = 100.0 * (rng.random(nspots)-0.5)
    z =  10.0 * (rng.random(nspots)-0.5)
    pists = rng.random(nspots)

    tstart = perf_counter()
    phase = np.zeros((slm.resolution, slm.resolution))
    phase[slm.pupil_idx] = executor.rs(x,y,z,pists)
    delta = perf_counter() - tstart

    print(f'rs algorithm (NSPOTS={nspots}): {delta:.2g} s')
    plt.title("rs algorithm")
    plt.imshow(phase)
    plt.show()

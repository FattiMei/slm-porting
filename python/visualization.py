import dependency_manager
np = dependency_manager.dep('numpy')

from slm import SLM
from time import perf_counter
import argparse
import matplotlib.pyplot as plt


def parse_command_line():
    parser = argparse.ArgumentParser(
        prog='visualization',
        description='Try a backend implementation of the `rs` algorithm'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='seed to initialize the RNG'
    )

    parser.add_argument(
        '--backend',
        type=str,
        help='Select the implementation (cpu, gpu, different ones)'
    )

    return parser.parse_args()


def validate_arguments(args):
    seed = args.seed
    backend = 'numpy' if args.backend is None else args.backend

    return seed, backend


if __name__ == '__main__':
    seed, requested_backend = validate_arguments(
        parse_command_line()
    )
    slm = SLM.get_standard_slm()
    available_backends = dependency_manager.get_available_backends()

    try:
        executor = available_backends[requested_backend].get_executor(slm)
    except KeyError:
        print(f'Backend "{requested_backend}" is not supported')
        print(f'Available backends:')

        for backend in available_backends.keys():
            print(f'  * {backend}')

        exit()

    rng = np.random.default_rng(seed)
    NPOINTS = 100
    x = 100.0 * (rng.random(NPOINTS)-0.5)
    y = 100.0 * (rng.random(NPOINTS)-0.5)
    z =  10.0 * (rng.random(NPOINTS)-0.5)
    pists = rng.random(NPOINTS)

    tstart = perf_counter()
    phase = np.zeros((slm.resolution, slm.resolution))
    phase[slm.pupil_idx] = executor.rs(x,y,z,pists)
    delta = perf_counter() - tstart

    print(f'rs algorithm (NPOINTS={NPOINTS}): {delta:.2g} s')
    plt.title("rs algorithm")
    plt.imshow(phase)
    plt.show()

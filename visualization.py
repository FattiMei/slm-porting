import argparse
import numpy as np
import matplotlib.pyplot as plt

from slmporting.core.slm import SLM
from slmporting.core.array import Array
from slmporting.impl.numpy import IMPLS


def parse_command_line():
    parser = argparse.ArgumentParser(
        prog='visualization',
        description='Try a backend implementation of the `rs` algorithm'
    )

    parser.add_argument('--seed', type=int, help='seed to initialize the RNG', default=42)
    parser.add_argument( '--nspots', type=int, help='The number of points to be generated', default=100)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_command_line()
    seed = args.seed
    nspots = args.nspots
    slm = SLM.get_standard_slm()

    rng = np.random.default_rng(seed)
    x = Array(100.0 * (rng.random(nspots)-0.5))
    y = Array(100.0 * (rng.random(nspots)-0.5))
    z = Array( 10.0 * (rng.random(nspots)-0.5))
    pists = Array(rng.random(nspots))

    impl = IMPLS[0]()
    print(impl.code)
    result, times = impl(x, y, z, pists, slm)
    phase = np.zeros((slm.resolution, slm.resolution))
    phase[slm.pupil_idx] = result

    print(f'rs algorithm (NSPOTS={nspots}): compile time {impl.comptime:.2g} s, transfer time {times.transfer_time:.2g} s, compute time {times.compute_time:.2g} s')
    plt.title("rs algorithm")
    plt.imshow(phase)
    plt.show()

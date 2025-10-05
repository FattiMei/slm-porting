import argparse
import numpy as np
import itertools
import matplotlib.pyplot as plt

from slmporting.core.slm import SLM
from slmporting.core.array import AlignedArray
import slmporting.impl.jax
import slmporting.impl.numpy
import slmporting.impl.torch


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
    x = AlignedArray(100.0 * (rng.random(nspots)-0.5))
    y = AlignedArray(100.0 * (rng.random(nspots)-0.5))
    z = AlignedArray( 10.0 * (rng.random(nspots)-0.5))
    pists = AlignedArray(rng.random(nspots))

    IMPLS = itertools.chain(
        slmporting.impl.jax.IMPLS,
        slmporting.impl.numpy.IMPLS,
        slmporting.impl.torch.IMPLS
    )

    for blueprint in IMPLS:
        impl = blueprint()
        result, times = impl(x, y, z, pists, slm)
        result, times = impl(x, y, z, pists, slm)
        phase = np.zeros((slm.resolution, slm.resolution))
        phase[slm.pupil_idx] = result

        print(f'rs algorithm (NSPOTS={nspots}, {impl.backend}): compile time {impl.comptime:.2g} s, in_transfer time {times.in_transfer_time:.2g} s, out_transfer_time: {times.out_transfer_time:.2g}, compute time {times.compute_time:.2g} s')

    plt.title("rs algorithm")
    plt.imshow(phase)
    plt.show()

import argparse
import itertools

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import numpy as np

from slmporting.core.slm import SLM
from slmporting.core.array import Array
from slmporting.core.types import Backend, Device, DType
import slmporting.impl.impl_jax
import slmporting.impl.impl_numpy
import slmporting.impl.impl_torch


def key_to_int(key):
    """Combine two uint32 values into a single 64-bit integer seed."""
    return int(key[0]) << 32 | int(key[1])


def int_to_key(seed):
    """Recreate the PRNGKey from the integer seed."""
    upper = (seed >> 32) & 0xFFFFFFFF
    lower = seed & 0xFFFFFFFF
    return jnp.array([upper, lower], dtype=jnp.uint32)


def parse_command_line():
    parser = argparse.ArgumentParser(
        prog='verification',
        description='test the correctness of the implementations'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='seed to initialize the RNG',
        default=42
    )

    parser.add_argument(
        '--nruns',
        type=int,
        help='how many experiments to run',
        default=1
    )

    parser.add_argument(
        '--nspots',
        type=int,
        help='number of points to be generated',
        default=100
    )

    return parser.parse_args()


def error_function(x, y):
    # we are dealing with a number that lives in [0, 2π]
    # and wraps, so 2π-ε and ε are quite close
    #
    # for now we will use the standard error
    return np.max(np.abs(x-y))


if __name__ == '__main__':
    args = parse_command_line()
    nspots = args.nspots
    key = int_to_key(args.seed)

    all_implementations = [
        impl()
        for impl in itertools.chain(
            slmporting.impl.impl_numpy.IMPLS,
            slmporting.impl.impl_jax.IMPLS,
            slmporting.impl.impl_torch.IMPLS
        )
    ]

    slm = SLM.get_standard_slm()

    print('algorithm,name,backend,device,dtype,key,abs_err,total_time')

    for n in range(args.nruns):
        # subkey is the key to export in the user report
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, 4)

        # TODO: to suppress the jax alignment warnings we must reimplement
        # the `AlignedArray` class
        x     = Array(jax.random.uniform(subkeys[0], nspots, minval=-50.0, maxval=50.0))
        y     = Array(jax.random.uniform(subkeys[1], nspots, minval=-50.0, maxval=50.0))
        z     = Array(jax.random.uniform(subkeys[2], nspots, minval=-5.0, maxval=5.0))
        pists = Array(jax.random.uniform(subkeys[3], nspots, minval=0.0, maxval=1.0))

        reference_impl = all_implementations[0]
        reference_phase, _ = reference_impl(x, y, z, pists, slm)
        for impl in all_implementations:
            assert(reference_impl.algorithm == impl.algorithm)
            alternative_phase, times = impl(x, y, z, pists, slm)

            error = error_function(reference_phase, alternative_phase)
            total_time = times.transfer_time + times.compute_time

            print(','.join(map(str, (
                reference_impl.algorithm,
                impl.name,
                impl.backend,
                Device.CPU,
                DType.fp64,
                key_to_int(key),
                error,
                total_time
            ))))

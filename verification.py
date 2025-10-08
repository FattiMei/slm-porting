import argparse
import itertools

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

from slmporting.core.slm import SLM
from slmporting.core.array import Array
import slmporting.impl.jax
import slmporting.impl.numpy
import slmporting.impl.torch


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


if __name__ == '__main__':
    args = parse_command_line()
    nspots = args.nspots
    key = int_to_key(args.seed)

    all_implementations = [
        impl()
        for impl in itertools.chain(
            slmporting.impl.numpy.IMPLS,
            slmporting.impl.jax.IMPLS,
            slmporting.impl.torch.IMPLS
        )
    ]

    slm = SLM.get_standard_slm()

    print('algorithm,name,backend,device,dtype,key,abs_err,rel_err')

    for n in range(args.nruns):
        # subkey is the key to export in the user report
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, 4)

        x     = Array(jax.random.uniform(subkeys[0], nspots, minval=-50.0, maxval=50.0))
        y     = Array(jax.random.uniform(subkeys[1], nspots, minval=-50.0, maxval=50.0))
        z     = Array(jax.random.uniform(subkeys[2], nspots, minval=-5.0, maxval=5.0))
        pists = Array(jax.random.uniform(subkeys[3], nspots, minval=0.0, maxval=1.0))

        # reference = all_implementations[0](x, y, z, pists, slm)
        print(key_to_int(key))

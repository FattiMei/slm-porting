import argparse
from slm.common.loader import load
np = dependency_manager.dep('numpy')


def parse_command_line():
    parser = argparse.ArgumentParser(
        prog='verification',
        description='Confront multiple backends on the same input'
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
        help='how many times to perform the comparison',
        default=10
    )

    parser.add_argument(
        '--reference',
        type=str,
        help='which implementation you need to test against'
    )

    parser.add_argument(
        '--backends',
        type=str,
        help='which implementation you need to test against',
        nargs='+'
    )

    return parser.parse_args()


def print_available_backends(backends):
    print(f'Available backends:')
    for backend in available_backends:
        print(f'  * {backend}')


if __name__ == '__main__':
    # it's quite hard at the moment to select
    # the precision in which to carry the computation
    args = parse_command_line()
    seed = args.seed
    backends = dependency_manager.get_available_backends()

    # absolutely AWFUL logic, needs refactoring
    reference_backend = args.reference
    requested_backends = set(args.backends)
    available_backends = set(backends.keys())

    orphan_backends = requested_backends.difference(available_backends)
    planned_backends = requested_backends.intersection(available_backends)

    print(reference_backend, planned_backends)
    if reference_backend not in planned_backends:
        print(f'"{reference_backend}" is not available to be used as reference')
        print_available_backends(backends)
        exit()

    if len(orphan_backends) > 0:
        print('Warning: some requested backends were left out')
        for b in orphan_backends:
            print(f'  * {b}')

    if len(planned_backends) == 0:
        print('No available backend matches your requirements')
        print_available_backends(backends)

    backends = {name: backends[name] for name in planned_backends}

    print(planned_backends)

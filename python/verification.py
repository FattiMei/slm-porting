import argparse
from typing import NamedTuple
import dependency_manager
np = dependency_manager.dep('numpy')


def parse_command_line():
    parser = argparse.ArgumentParser(
        prog='verification',
        description='Confront multiple backends on the same input'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='seed to initialize the RNG'
    )

    parser.add_argument(
        '--nruns', 
        type=int,
        help='how many times to perform the comparison'
    )

    parser.add_argument(
        '--algorithms',
        type=str,
        nargs='+',
        help='which algorithm to test'
    )

    return parser.parse_args()


def validate_arguments(args):
    seed = args.seed
    nruns = 1 if args.nruns is None else args.nruns
    algorithms = args.algorithms

    if nruns < 1 or nruns > 100:
        print("Let's keep the number or runs between 1 and 100")
        exit()

    return {
        'seed': seed,
        'nruns': nruns
    }


if __name__ == '__main__':
    # from the command line arguments one could also
    # select the backends to be tested, completed with the
    # precision in which to carry the computation
    args = validate_arguments(
        parse_command_line()
    )

    rng = np.random.default_rng(args['seed'])

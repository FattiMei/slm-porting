import pathlib
import pkgutil
import unittest
import importlib


# centralized dependency loading. This way I guarantee that
# every dependency is loaded with the appropriate configuration,
# without repetitions
#
# this is particularly true for `jax` because it doesn't natively
# support float64 dtypes
def load(name: str):
    module = None

    try:
        module = importlib.import_module(name)

        if name == 'jax':
            module.config.update('jax_enable_x64', True)

            # the number of CPU cores should be queried from the system
            # or from an environment variable
            #
            # this step is necessary otherwise the cpu implementation
            # would assume only one core
            module.config.update('jax_num_cpu_devices', 8)

    # this is the only exception I'm expecting
    except ImportError:
        raise ImportError

    return module


# this function is used by the programs in the root folder. It is meant to
# selectively include implementation files when their respective dependencies
# are satisfied.
#
# I have designed the folder structure for the following requirements:
#   * in the root folder I want only artifacts usable by the end user (peer reviewers)
#   * the implementation files (which could grow up to 10 files) should be in a separate folder
#   * the common files should not pollute the root folder
#
# with this folder structure, an implementation must include files that are not in its directory
# this means that one needs to launch the implementation file from the root directory if necessary
# that is a little price to pay for a greater cause.
def get_available_backends() -> dict:
    root_path = pathlib.Path(__file__).parent.parent
    impl_path = root_path / 'impl'

    print(root_path)

    backends = {}
    for _, name, _ in pkgutil.iter_modules([str(impl_path)]):
        print(name)
        backends[name] = importlib.import_module(f'slm.impl.{name}')

        # try:
        #     backends[name] = importlib.import_module(f'impl.{name}')
        # except ImportError:
        #     pass

    return backends


def print_available_backends(backends=None):
    if backends is None:
        backends = get_available_backends()

    print('Available backends:')
    for b in backends.keys():
        print(f'  * b')


class TestDependencyLoad(unittest.TestCase):
    def test_load(self):
        _ = load('numpy')

        with self.assertRaises(ImportError):
            _ = load('aaaaaaa')

    def test_backend_discovery(self):
        backends = get_available_backends()
        self.assertTrue('numpy' in backends)


if __name__ == '__main__':
    unittest.main()

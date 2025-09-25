import pathlib
import pkgutil
import importlib
import unittest


# this single module import hopefully improves the loading
# time for some programs
def dep(name: str):
    module = None

    try:
        module = importlib.import_module(name)

        if name == 'jax':
            module.config.update('jax_enable_x64', True)

    except ImportError:
        raise ImportError

    return module


# I can't import the backends in the main scope
# because they import this file, so there are circular dependencies
def get_available_backends() -> dict:
    backends = {}
    root_path = pathlib.Path(__file__).parent
    impl_path = root_path / 'impl'

    for _, name, _ in pkgutil.iter_modules([str(impl_path)]):
        try:
            backends[name] = importlib.import_module(f'impl.{name}')
        except ImportError:
            pass

    return backends


class TestDependencyManager(unittest.TestCase):
    def test_backend_discovery(self):
        backends = get_available_backends()
        self.assertTrue('numpy' in backends)


if __name__ == '__main__':
    unittest.main()

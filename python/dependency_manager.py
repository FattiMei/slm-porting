import importlib


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
    raise NotImplemented

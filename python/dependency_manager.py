import importlib


DEPS = {}

for name in ['numpy', 'jax', 'torch', 'triton']:
    try:
        DEPS[name] = importlib.import_module(name)
    except ImportError:
        pass

import numpy as np
import jax
import jax.numpy as jnp
import torch
import pytest

from slmporting.core.array import Array, CachedArray, numpy_dtype_map
from slmporting.core.types import Backend, Device, DType


all_backends = [Backend.NUMPY, Backend.JAX, Backend.TORCH]
all_dtypes = [DType.fp16, DType.fp32, DType.fp64]


def ones(size: int, backend: Backend):
    if backend == Backend.NUMPY:
        return np.ones(size)

    elif backend == Backend.JAX:
        return jnp.ones(size)

    elif backend == Backend.TORCH:
        return torch.ones(size)

    else:
        assert(False)


@pytest.mark.parametrize("backend", all_backends)
def test_array_creation(backend: Backend):
    data = ones(100, backend)
    arr = Array(data)


@pytest.mark.parametrize("source", all_backends)
@pytest.mark.parametrize("dest", all_backends)
@pytest.mark.parametrize("dtype", all_dtypes)
def test_cpu_conversion(source: Backend, dest: Backend, dtype: DType):
    data = ones(100, source)
    arr = Array(data)

    converted = arr.convert_to(
        backend = dest,
        device = Device.CPU,
        dtype = dtype
    )



@pytest.mark.parametrize("backend", all_backends)
@pytest.mark.parametrize("dtype", all_dtypes)
def test_caching(backend: Backend, dtype: DType):
    arr = CachedArray(np.random.rand(100))

    first = arr.convert_to(
        backend = backend,
        device = Device.CPU,
        dtype = dtype
    )

    # this second time the array has already been
    # cached, so there isn't a copy
    second = arr.convert_to(
        backend = backend,
        device = Device.CPU,
        dtype = dtype
    )

    # one would have to test that the two arrays
    # share the same memory
    #
    # they are not anymore the same object because
    # they are build on the go from a previously
    # allocated memory

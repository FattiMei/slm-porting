import numpy as np
import pytest

from slmporting.core.array import Array, CachedArray, numpy_dtype_map
from slmporting.core.types import Backend, Locality, DType


@pytest.mark.parametrize("size", [10, 100, 1000])
def test_array_creation(size: int):
    arr = Array(np.zeros(size))


@pytest.mark.parametrize("dtype", [DType.fp16, DType.fp32, DType.fp64])
def test_numpy_conversion(dtype: DType):
    arr = Array(np.zeros(100))
    converted = arr.convert_to(
        backend = Backend.NUMPY,
        locality = Locality.CPU,
        dtype = dtype
    )

    assert(type(converted) == np.ndarray)
    assert(converted.dtype == numpy_dtype_map[dtype])


@pytest.mark.parametrize("backend", [Backend.NUMPY, Backend.JAX, Backend.TORCH])
@pytest.mark.parametrize("dtype", [DType.fp16, DType.fp32, DType.fp64])
def test_caching(backend: Backend, dtype: DType):
    arr = CachedArray(np.random.rand(100))

    first = arr.convert_to(
        backend = backend,
        locality = Locality.CPU,
        dtype = dtype
    )

    # this second time the array has already been
    # cached, so there isn't a copy
    second = arr.convert_to(
        backend = backend,
        locality = Locality.CPU,
        dtype = dtype
    )

    # one would have to test that the two arrays
    # share the same memory
    #
    # they are not anymore the same object because
    # they are build on the go from a previously
    # allocated memory

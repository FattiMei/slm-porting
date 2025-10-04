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

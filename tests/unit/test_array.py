import numpy as np
import jax
import jax.numpy as jnp
import torch
import pytest

from slmporting.core.array import Array, CachedArray, numpy_dtype_map, jax_dtype_map, torch_dtype_map, jax_device_map
from slmporting.core.types import Backend, Device, DType
from slmporting.utils.device_discovery import get_available_devices


all_backends = [Backend.NUMPY, Backend.JAX, Backend.TORCH]
all_devices = get_available_devices()
all_dtypes = [DType.fp16, DType.fp32, DType.fp64]


backend_type_map = {
    Backend.NUMPY: np.ndarray,
    Backend.JAX: jax.Array,
    Backend.TORCH: torch.Tensor
}


def ones(size: int, backend: Backend, device: Device = Device.CPU, dtype: DType = DType.fp64):
    if backend == Backend.NUMPY:
        assert(device == Device.CPU)
        return np.ones(
            size,
            dtype=numpy_dtype_map[dtype]
        )

    elif backend == Backend.JAX:
        return jnp.ones(
            size,
            device=jax_device_map[device],
            dtype=jax_dtype_map[dtype]
        )

    elif backend == Backend.TORCH:
        arr = torch.ones(
            size,
            dtype=torch_dtype_map[dtype]
        )

        if device == Device.GPU:
            arr = arr.to('cuda')

        return arr

    else:
        assert(False)


@pytest.mark.parametrize("backend", all_backends)
@pytest.mark.parametrize("device", all_devices)
@pytest.mark.parametrize("dtype", all_dtypes)
def test_array_creation(backend: Backend, device: Device, dtype: DType):
    data = ones(100, backend, device, dtype)
    arr = Array(data)


@pytest.mark.parametrize("source", all_backends)
@pytest.mark.parametrize("dest", all_backends)
@pytest.mark.parametrize("dtype", all_dtypes)
def test_cpu_conversion(source: Backend, dest: Backend, dtype: DType):
    data = ones(100, source, device = Device.CPU)
    arr = Array(data)

    converted = arr.convert_to(
        backend = dest,
        device = Device.CPU,
        dtype = dtype
    )

    assert(isinstance(converted, backend_type_map[dest]))


@pytest.mark.parametrize("source_backend", all_backends)
@pytest.mark.parametrize("source_device",  all_devices)
@pytest.mark.parametrize("source_dtype",   all_dtypes)
@pytest.mark.parametrize("dest_backend", all_backends)
@pytest.mark.parametrize("dest_device",  all_devices)
@pytest.mark.parametrize("dest_dtype",   all_dtypes)
def test_any_to_any_conversion(
    source_backend: Backend,
    source_device: Device,
    source_dtype: DType,
    dest_backend: Backend,
    dest_device: Device,
    dest_dtype: DType):

    arr = Array(ones(
        100, 
        backend = source_backend,
        device = source_device,
        dtype = source_dtype
    ))

    converted = arr.convert_to(
        backend = dest_backend,
        device = dest_device,
        dtype = dest_dtype
    )

    assert(isinstance(converted, backend_type_map[dest_backend]))


@pytest.mark.parametrize("backend", all_backends)
@pytest.mark.parametrize("dtype", all_dtypes)
def test_caching(backend: Backend, dtype: DType):
    arr = CachedArray(Array(np.random.rand(100)))

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
    assert(isinstance(first, backend_type_map[backend]))
    assert(isinstance(second, backend_type_map[backend]))


# We need to make sure that the transfers that are supposed to be zero copy
# are in fact zero copy.
#
# With CachedArray, some bad implementations could still provide the expected
# performance because the results are cached.

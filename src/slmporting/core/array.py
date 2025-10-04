import jax
import jax.numpy as jnp
import torch
import torch.utils.dlpack
import numpy as np

from slmporting.core.types import Backend, Locality, DType


numpy_dtype_map = {
    DType.fp16:  np.float16,
    DType.fp32:  np.float32,
    DType.fp64:  np.float64,
}


jax_dtype_map = {
    DType.fp16:  'float16',
    DType.fp32:  'float32',
    DType.fp64:  'float64',
}


jax_device_map = {
    Locality.CPU: 'cpu',
    Locality.GPU: 'gpu'
}


torch_dtype_map = {
    DType.fp16:  torch.float16,
    DType.fp32:  torch.float32,
    DType.fp64:  torch.float64
}


torch_device_map = {
    Locality.CPU: 'cpu',
    Locality.GPU: 'cuda'
}


def inverse_search_dict(dtype_map, value):
    for key in dtype_map.keys():
        if dtype_map[key] == value:
            return key


class Array:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            backend = Backend.NUMPY
            locality = Locality.CPU
            dtype = inverse_search_dict(numpy_dtype_map, data.dtype)

        elif isinstance(data, jax.numpy.array):
            backend = Backend.JAX
            locality = Locality.CPU if data.device().platform == 'cpu' else Locality.GPU
            dtype = inverse_search_dict(jax_dtype_map, data.dtype)

        elif isinstance(data, torch.tensor):
            backend = Backend.TORCH
            locality = Locality.CPU if data.device.type == 'cpu' else Locality.GPU
            dtype = inverse_search_dict(torch_dtype_map, data.dtype)

        else:
            assert(False)

        self.data = data
        self.backend = backend
        self.locality = locality
        self.dtype = dtype

    '''
    returns an array of the requested type (zero-copy if possible)
    '''
    def convert_to(self, backend: Backend, locality: Locality, dtype: DType):
        if backend == Backend.NUMPY:
            assert(locality == Locality.CPU)
            result = np.array(
                np.from_dlpack(self.data),
                dtype = numpy_dtype_map[dtype]
            )

        elif backend == Backend.JAX:
            result = jnp.array(
                jnp.from_dlpack(self.data, device=jax_device_map[locality]),
                dtype = jax_dtype_map[dtype]
            )

        elif backend == Backend.TORCH:
            result = torch.from_dlpack(self.data).to(
                device = torch_device_map[locality],
                dtype = torch_dtype_map[dtype]
            )

        else:
            assert(False)

        return result


class CachedArray:
    def __init__(self, data):
        self.arr = Array(data)
        self.cache = {
            Locality.CPU: {},
            Locality.GPU: {}
        }

    def convert_to(
        self,
        backend: Backend,
        locality: Locality = Locality.CPU, 
        dtype: DType = DType.fp64
    ):
        if dtype not in self.cache[locality]:
            self.cache[locality][dtype] = self.arr.convert_to(
                backend = Backend.TORCH,
                locality = locality,
                dtype = dtype
            )

        return self.cache[locality][dtype]

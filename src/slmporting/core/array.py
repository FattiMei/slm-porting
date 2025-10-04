import jax
import jax.numpy as jnp
import torch
import torch.utils.dlpack
import numpy as np

from slmporting.core.types import Backend, Device, DType


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
    Device.CPU: jax.devices(backend='cpu')[0],
    Device.GPU: 'gpu'
}


torch_dtype_map = {
    DType.fp16:  torch.float16,
    DType.fp32:  torch.float32,
    DType.fp64:  torch.float64
}


torch_device_map = {
    Device.CPU: 'cpu',
    Device.GPU: 'cuda'
}


def inverse_search_dict(dtype_map, value):
    for key in dtype_map.keys():
        if dtype_map[key] == value:
            return key


class Array:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            backend = Backend.NUMPY
            device = Device.CPU
            dtype = inverse_search_dict(numpy_dtype_map, data.dtype)

        elif isinstance(data, jax.Array):
            backend = Backend.JAX
            device = Device.CPU if data.device.platform == 'cpu' else Device.GPU
            dtype = inverse_search_dict(jax_dtype_map, data.dtype)

        elif isinstance(data, torch.Tensor):
            backend = Backend.TORCH
            device = Device.CPU if data.device.type == 'cpu' else Device.GPU
            dtype = inverse_search_dict(torch_dtype_map, data.dtype)

        else:
            assert(False)

        self.data = data
        self.backend = backend
        self.device = device
        self.dtype = dtype

    '''
    returns an array of the requested type (zero-copy if possible)
    '''
    def convert_to(self, backend: Backend, device: Device, dtype: DType):
        if backend == Backend.NUMPY:
            assert(device == Device.CPU)
            result = np.array(
                np.from_dlpack(self.data),
                dtype = numpy_dtype_map[dtype]
            )

        elif backend == Backend.JAX:
            result = jnp.from_dlpack(self.data, device=jax_device_map[device])

        elif backend == Backend.TORCH:
            result = torch.from_dlpack(self.data).to(
                device = torch_device_map[device],
                dtype = torch_dtype_map[dtype]
            )

        else:
            assert(False)

        return result


class CachedArray:
    def __init__(self, data):
        self.arr = Array(data)
        self.cache = {
            Device.CPU: {},
            Device.GPU: {}
        }

    def convert_to(
        self,
        backend: Backend,
        device: Device = Device.CPU, 
        dtype: DType = DType.fp64
    ):
        if dtype not in self.cache[device]:
            self.cache[device][dtype] = Array(
                self.arr.convert_to(
                    backend = Backend.TORCH,
                    device = device,
                    dtype = dtype
                )
            )

        return self.cache[device][dtype].convert_to(
            backend,
            device,
            dtype
        )

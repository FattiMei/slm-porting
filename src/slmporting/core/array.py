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
}

try:
    jax_device_map[Device.GPU] = jax.devices(backend='gpu')[0]
except RuntimeError:
    pass


torch_dtype_map = {
    DType.fp16:  torch.float16,
    DType.fp32:  torch.float32,
    DType.fp64:  torch.float64
}


torch_device_map = {
    Device.CPU: 'cpu',
    Device.GPU: 'cuda'
}


'''
This class manages an internally allocated torch.Tensor and
its conversion to every possible backend, device and dtype

The choice to use only a torch.Tensor is simpler to understand
and to control
'''
class Array:
    def __init__(self, data):
        self.data = torch.from_dlpack(data)
        self.device = Device.CPU if self.data.device.type == 'cpu' else Device.GPU

    '''
    returns an array of the requested type (zero-copy if possible)
    '''
    def convert_to(self, backend: Backend, device: Device, dtype: DType):
        converted = self.data.to(
            device = torch_device_map[device],
            dtype = torch_dtype_map[dtype],
            copy = False
        )

        if backend == Backend.NUMPY:
            assert(device == Device.CPU)
            result = converted.numpy()

        elif backend == Backend.JAX:
            result = jnp.from_dlpack(converted)

        elif backend == Backend.TORCH:
            result = converted

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

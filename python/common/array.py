import numpy as np
import torch
import jax

from enum import Enum


class Backend(Enum):
    NUMPY = 0
    JAX   = 1
    TORCH = 2


class Locality(Enum):
    CPU = 0
    GPU = 1


class DType(Enum):
    fp8   = 1
    fp16  = 2
    fp32  = 4
    fp64  = 8
    fp128 = 16


numpy_dtype_map = {
    DType.fp16:  np.float16,
    DType.fp32:  np.float32,
    DType.fp64:  np.float64,
    DType.fp128: np.float128,
}


jax_dtype_map = {
    DType.fp16:  'float16',
    DType.fp32:  'float32',
    DType.fp64:  'float64',
    DType.fp128: 'float128',
}


class Array:
    def __init__(self, data: np.ndarray):
        self.data = data

    def convert_to(
        self,
        backend: Backend,
        locality: Locality = Locality.CPU, 
        dtype: DType = DType.fp64
    ):
        if backend == Backend.NUMPY:
            assert(locality = Locality.CPU)
            return np.array(self.data, dtype=numpy_dtype_map[dtype])

        elif: backend == Backend.JAX:
            if locality == Locality.CPU:
                return jnp.array(self.data, dtype=jax_dtype_map[dtype])

            elif locality == Locality.GPU:
                # still in progress because I can't test it on my machine
                pass

            else:
                assert(False)

        elif: backend == Backend.TORCH:
            if locality == Locality.CPU:
                return torch.from_numpy(self.data)

            elif locality == Locality.GPU:
                # still in progress because I can't test it on my machine
                pass

            else:
                assert(False)

        else:
            assert(False)


class CachedArray(Array):
    def __init__(self, data: np.ndarray):
        super().__init__(data)
        self.cache = {
            Locality.CPU: {},
            Locality.GPU: {}
        }

    def convert_to(
        backend: Backend,
        locality: Locality = Locality.CPU, 
        dtype: DType = DType.fp64
    ):
        # logica abbastanza normale, non la sviluppo adesso
        # perché perderei solo tempo.
        #
        # basti pensare che questo è in grado di cachare il dato (compreso il suo dtype)
        # ed è molto comodo per le implementazioni gpu
        pass

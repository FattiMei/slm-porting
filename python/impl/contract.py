import inspect
from enum import Enum
from typing import Callable, NamedTuple, TypeVar
from time import perf_counter


class FloatType(Enum):
    fp16 = 2
    fp32 = 4
    fp64 = 8


class ProfileInfo(NamedTuple):
    compiletime: float
    runtime: float


T = TypeVar('T')
U = TypeVar('U')
ConversionFunction = Callable[[T, FloatType], U]


def implementation(
    algorithm_class: str,
    backend_name: str,
    conversion_from_numpy_to_native: ConversionFunction,
    conversion_from_native_to_numpy: ConversionFunction,
    compilation_function = lambda x: x,
    description: str = ''
):
    def deco(fn):
        impl_name = fn.__name__
        compile_fcn = compilation_function

        class Implementation:
            alg_class = algorithm_class
            backend = backend_name
            name = impl_name
            signature = inspect.signature(fn)
            desc = description

            def __init__(self, dtype: FloatType):
                self.dtype = dtype

                t0 = perf_counter()
                self.fn = compile_fcn(fn)
                t1 = perf_counter()
                self.compile_time = t1 - t0

            def call_unsafe(self, x, y, z, pists, **kwargs):
                return self.fn(x, y, z, pists, **kwargs)

            def call(self, x, y, z, pists, **kwargs):
                t0 = perf_counter()
                x = conversion_from_numpy_to_native(x, self.dtype)
                y = conversion_from_numpy_to_native(y, self.dtype)
                z = conversion_from_numpy_to_native(z, self.dtype)
                pists = conversion_from_numpy_to_native(pists, self.dtype)
                t1 = perf_counter()

                result = self.call_unsafe(x, y, z, pists, **kwargs)

                t2 = perf_counter()
                result = conversion_from_native_to_numpy(result)
                t3 = perf_counter()

                return result, NamedTuple(
                    transfer_time = (t1-t0) + (t3-t2),
                    compute_time = t2-t1
                )

        return Implementation

    return deco

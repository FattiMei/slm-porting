import inspect
from time import perf_counter
from typing import NamedTuple


def implementation(
    algorithm_class: str,
    backend_name: str,
    conversion_from_numpy_to_native,
    conversion_from_native_to_numpy,
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

            def __init__(self):
                t0 = perf_counter()
                self.fn = compile_fcn(fn)
                t1 = perf_counter()
                self.compile_time = t1 - t0

            def call_unsafe(self, x, y, z, pists, **kwargs):
                return self.fn(x, y, z, pists, **kwargs)

            def call(self, x, y, z, pists, **kwargs):
                t0 = perf_counter()
                x = conversion_from_numpy_to_native(x)
                y = conversion_from_numpy_to_native(y)
                z = conversion_from_numpy_to_native(z)
                pists = conversion_from_numpy_to_native(pists)
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

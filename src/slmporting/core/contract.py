import time
import numpy as np
import inspect
import itertools
from collections.abc import Iterable

from slmporting.core.slm import SLM
from slmporting.core.types import Algorithm, Backend, Device, DType, ProfileInfo, Tensor


algorithm_signature_map = {
    Algorithm.RS: ['x', 'y', 'z', 'pists'],
    Algorithm.GS: ['x', 'y', 'z', 'pists', 'iter'],
}


def build_interface(signature, target_signature, backend: Backend):
    arguments = [p for p in signature.parameters]

    if len(arguments) < len(target_signature):
        raise ValueError(f'Too few arguments to be compatible with function signature: requested {len(target_signature)}, got {len(signature)}')

    if arguments[:len(target_signature)] != target_signature:
        raise ValueError(f'{signature} is not compatible with {target_signature}. I need exact matches in argument names')

    tensor_arguments = [
        p
        for p in signature.parameters
        if signature.parameters[p].annotation == Tensor
    ]

    foreign_arguments = [
        p
        for p in arguments
        if p not in target_signature
    ]

    acc = []

    header = f'def __call__(self, {", ".join(target_signature)}, slm: SLM, device: Device = Device.CPU, dtype: DType = DType.fp64):'
    acc.append('# get reference time')
    acc.append('t0 = time.perf_counter()\n')

    acc.append('# extract foreign arguments from `slm`')
    for p in foreign_arguments:
        acc.append(f'{p} = slm.{p}')

    acc.append('# convert all tensor arguments to proper backend')
    for p in tensor_arguments:
        acc.append(f'{p} = {p}.convert_to({backend}, device, dtype)')
    acc.append('')

    acc.append('t1 = time.perf_counter()\n')

    function_call = 'self.fn(' + ','.join(p for p in arguments) + ')'
    acc.append('# call function with the original signature')
    acc.append(f'result = {function_call}')

    acc.append('t2 = time.perf_counter()')
    acc.append('# cast result to the usual numpy array')
    acc.append('result = np.array(result, dtype=np.float64)')
    acc.append('t3 = time.perf_counter()\n')

    acc.append('# discriminate between computation time and transfer time')
    acc.append('return result, ProfileInfo(in_transfer_time=(t1-t0), out_transfer_time=(t3-t2), compute_time=(t2-t1))')

    indended_code = map(
        lambda line: '    ' + line,
        acc
    )
    complete_code = itertools.chain([header], indended_code)
    return '\n'.join(complete_code)



def impl(algorithm: Algorithm, backend: Backend, device: Device, compiler = (lambda x: x), description: str = None):
    local_algorithm   = algorithm
    local_backend     = backend
    local_compiler    = compiler
    local_description = description

    local_devices = []
    if isinstance(device, Device):
        local_devices.append(device)
    elif isinstance(device, Iterable):
        for d in device:
            assert(isinstance(d, Device))
            local_devices.append(d)
    else:
        assert(False)

    def decorator(fn):
        signature = inspect.signature(fn)
        local_code = build_interface(
            signature,
            algorithm_signature_map[algorithm],
            backend
        )
        local_vars = {
            'np': np,
            'time': time,
            'Backend': Backend,
            'DType': DType,
            'Device': Device,
            'SLM': SLM,
            'ProfileInfo': ProfileInfo
        }
        exec(local_code, local_vars)

        class Implementation:
            name        = fn.__name__
            algorithm   = local_algorithm
            backend     = local_backend
            compiler    = local_compiler
            description = local_description
            devices     = local_devices
            code        = local_code

            __call__ = local_vars['__call__']

            def __init__(self):
                t0 = time.perf_counter()
                self.fn = local_compiler(fn)
                t1 = time.perf_counter()
                self.comptime = t1 - t0

        return Implementation

    return decorator

import time
import inspect
import itertools
import numpy as np
from collections.abc import Iterable

from slm import SLM
from traits import Algorithm, Backend, DType, Locality, ProfileInfo


algorithm_signature_map = {
    Algorithm.RS: ['x', 'y', 'z', 'pists'],
    Algorithm.GS: ['x', 'y', 'z', 'pists', 'iter'],
}


class Tensor:
    def __repr__(self):
        return 'Tensor'


def _build_interface(signature, target_signature, backend: Backend):
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

    header = f'def __call__(self, {", ".join(target_signature)}, slm: SLM, locality: Locality = Locality.CPU, dtype: DType = DType.fp64):'
    acc.append('# get reference time')
    acc.append('t0 = time.perf_counter()\n')

    acc.append('# extract foreign arguments from `slm`')
    for p in foreign_arguments:
        acc.append(f'{p} = slm.{p}')

    acc.append('# convert all tensor arguments to proper backend')
    for p in tensor_arguments:
        acc.append(f'{p} = {p}.convert_to({backend}, locality, dtype)')
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
    acc.append('return result, ProfileInfo(transfer_time=(t1-t0)+(t3-t2), compute_time=(t2-t1))')

    indended_code = map(
        lambda line: '    ' + line,
        acc
    )
    complete_code = itertools.chain([header], indended_code)
    return '\n'.join(complete_code)



def impl(algorithm: Algorithm, backend: Backend, locality: Locality, compiler = (lambda x: x), description: str = None):
    # it seems like we have to copy the arguments into local
    # variables before using them inside the class
    local_algorithm   = algorithm
    local_backend     = backend
    local_compiler    = compiler
    local_description = description

    local_locality = []
    if isinstance(locality, Locality):
        local_locality.append(locality)
    elif isinstance(locality, Iterable):
        for l in locality:
            assert(isinstance(l, Locality))
            local_locality.append(l)
    else:
        assert(False)

    def decorator(fn):
        signature = inspect.signature(fn)
        local_code = _build_interface(
            signature,
            algorithm_signature_map[algorithm],
            backend
        )
        local_vars = {
            'np': np,
            'time': time,
            'Backend': Backend,
            'DType': DType,
            'Locality': Locality,
            'ProfileInfo': ProfileInfo
        }
        exec(local_code, local_vars)

        class Implementation:
            # these are parameter used to filter the implementations
            # even before constructing the objects
            name        = fn.__name__
            algorithm   = local_algorithm
            backend     = local_backend
            compiler    = local_compiler
            description = local_description
            locality    = local_locality
            code        = local_code

            __call__ = local_vars['__call__']

            def __init__(self):
                t0 = time.perf_counter()
                self.fn = local_compiler(fn)
                t1 = time.perf_counter()
                self.comptime = t1 - t0

        return Implementation

    return decorator

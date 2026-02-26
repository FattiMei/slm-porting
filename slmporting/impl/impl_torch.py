import os
os.environ["CXXFLAGS"] = "-march=native -ffast-math"

import torch
import numpy as np
from functools import partial

from slmporting.core.types import Algorithm, Backend, Device, DType, Tensor
from slmporting.core.contract import impl


ε = None


torch_default_compiler = partial(
    torch.compile,
    mode='default',
    backend='inductor',
    dynamic=False,
    fullgraph=True
)

torch_optimized_compiler = partial(
    torch.compile,
    backend='inductor',
    dynamic=False,
    fullgraph=True,
    options = {
        'max_autotune': True,
        'use_fast_math': True,
        'aggressive_fusion': True,
        'cpp.enable_tiling_heuristics': True,
        'cpp.enable_unsafe_math_opt_flag': True,
    }
)


torch_compiler = torch_optimized_compiler


@impl(Algorithm.RS, Backend.TORCH, [Device.CPU, Device.GPU], compiler = torch_compiler, description = 'same code, just change math functions')
def rs(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    return torch.angle(
        torch.mean(
            torch.exp(
                1j * (
                    C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) +
                    C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] +
                    2*np.pi*pists[:,ε]
                )
            ),
            dim=0
        )
    )


@impl(Algorithm.RS, Backend.TORCH, [Device.CPU, Device.GPU], compiler = torch_compiler, description = 'remove complex operators, not supported in compilation')
def rs_no_complex(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    slm_p_phase = C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) + \
                  C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] + \
                  2*np.pi*pists[:,ε]

    avg_field = torch.vstack((
        torch.mean(torch.cos(slm_p_phase), dim=0),
        torch.mean(torch.sin(slm_p_phase), dim=0)
    ))

    return torch.arctan2(avg_field[1], avg_field[0])


@impl(Algorithm.RS, Backend.TORCH, [Device.CPU, Device.GPU], compiler = torch_compiler, description = 'using for loops')
def rs_for_loop(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    sines = torch.zeros_like(xx)
    cosines = torch.zeros_like(xx)

    for i in range(x.shape[0]):
        slm_p_phase = C1 * (x[i]*xx + y[i]*yy) + \
                      C2 * z[i] * (xx**2 + yy**2) + \
                      2*np.pi*pists[i]

        sines += torch.sin(slm_p_phase)
        cosines += torch.cos(slm_p_phase)

    return torch.arctan2(sines, cosines)


IMPLS = [rs, rs_no_complex, rs_for_loop]

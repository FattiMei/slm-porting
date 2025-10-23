import torch
import numpy as np

from slmporting.core.types import Algorithm, Backend, Device, DType, Tensor
from slmporting.core.contract import impl


ε = None


@impl(Algorithm.RS, Backend.TORCH, [Device.CPU, Device.GPU], compiler = torch.compile, description = 'same code, just change math functions')
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


@impl(Algorithm.RS, Backend.TORCH, [Device.CPU, Device.GPU], compiler = torch.compile, description = 'remove complex operators, not supported in compilation')
def rs_no_complex(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    slm_p_phase = C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) + \
                  C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] + \
                  2*np.pi*pists[:,ε]

    avg_field = torch.vstack((
        torch.mean(torch.cos(slm_p_phase), dim=0),
        torch.mean(torch.sin(slm_p_phase), dim=0)
    ))

    return torch.arctan2(avg_field[1], avg_field[0])


IMPLS = [rs, rs_no_complex]

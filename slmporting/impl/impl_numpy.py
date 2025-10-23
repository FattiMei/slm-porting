import numpy as np

from slmporting.core.types import Algorithm, Backend, Device, DType, Tensor
from slmporting.core.contract import impl


ε = np.newaxis


@impl(Algorithm.RS, Backend.NUMPY, [Device.CPU], description = 'array programming implementation')
def rs(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    return np.angle(
        np.mean(
            np.exp(
                1j * (
                    C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) +
                    C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] +
                    2*np.pi*pists[:,ε]
                )
            ),
            axis=0
        )
    )


@impl(Algorithm.RS, Backend.NUMPY, [Device.CPU], description = 'simulate complex numbers with real numbers')
def rs_no_complex(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    slm_p_phase = C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) + \
                  C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] + \
                  2*np.pi*pists[:,ε]

    avg_field = np.vstack((
        np.mean(np.cos(slm_p_phase), axis=0),
        np.mean(np.sin(slm_p_phase), axis=0)
    ))

    return np.arctan2(avg_field[1], avg_field[0])


@impl(Algorithm.RS, Backend.NUMPY, [Device.CPU], description = 'do not store the complete slm_p_phase')
def rs_manual_loop(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    avg_field_x = np.empty(xx.shape, xx.dtype)
    avg_field_y = np.empty(yy.shape, yy.dtype)

    for i in range(xx.shape[0]):
        p = C1 * (x*xx[i] + y*yy[i]) + \
            C2 * z * (xx[i]**2 + yy[i]**2) + \
            2*np.pi*pists

        avg_field_x[i] = np.mean(np.cos(p))
        avg_field_y[i] = np.mean(np.sin(p))

    return np.arctan2(avg_field_y, avg_field_x)


IMPLS = [rs, rs_no_complex]

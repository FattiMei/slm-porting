import numpy as np
from typing import NamedTuple
import units
from slm import SLM
from rng import Rng


# this is a clever way to improve notation,
# however it must be compatible with the future jax implementation
# I may change it with `None`
ε = np.newaxis


def rs(slm: SLM, x, y, z, rng: Rng) -> np.ndarray:
    # these are aliases for lightening the notation
    C1 = slm.C1
    C2 = slm.C2
    xx = slm.xx
    yy = slm.yy

    result = np.zeros((slm.resolution, slm.resolution))
    result[slm.pupil_idx] = np.angle(
        np.mean(
            np.exp(
                1j * (
                    C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) +
                    C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] +
                    2*np.pi*rng.sample(x.shape)[:,ε]
                )
            ),
            axis=0
        )
    )

    return result


class QualityMetrics(NamedTuple):
    efficiency: float
    uniformity: float
    variance:   float


'''
This function is useful to domain experts to characterize the quality
of the proposed algorithms.

It recomputes the `slm_p_phase` variable. I got to the conclusion that
this is the best choice given that memory accesses are generally slower
than computation.

    +-----------+---------------+--------------+
    |  policy   |   memory cost | compute cost |
    +-----------+---------------+--------------+
    | store     |    O(N*M)     |    O(1)      |
    | recompute |    O(N+M)     |   O(N*M)     |
    +-----------+---------------+--------------+
'''
def compute_quality_metrics(slm: SLM, x, y, z, phase) -> QualityMetrics:
    C1 = slm.C1
    C2 = slm.C2
    xx = slm.xx
    yy = slm.yy

    intensities = np.abs(
        np.mean(
            np.exp(
                1j * (
                    phase[slm.pupil_idx][ε,:] -
                    C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) +
                    C2 * z[:,ε] * (xx**2 + yy**2)[ε,:]
                )
            ),
            axis=1
        )
    )**2

    min = np.min(intensities)
    max = np.max(intensities)

    return QualityMetrics(
        efficiency = np.sum(intensities),
        uniformity = 1 - (max - min) / (max + min),
        variance   = np.sqrt(np.var(intensities)) / np.mean(intensities)
    )

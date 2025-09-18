import numpy as np
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

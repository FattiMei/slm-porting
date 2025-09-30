from slm.common.slm import SLM, QualityMetrics
from slm.common.executor import Executor
from slm.common.loader import load
np = load('numpy')


class NumpyExecutor(Executor):
    def __init__(self, slm: SLM):
        super().__init__(slm)

    def _convert_from_numpy_to_native(self, x: np.ndarray):
        return x

    def _convert_from_native_to_numpy(self, native):
        return native

    def _rs(self, x, y, z, pists):
        return rs_soa_pupil_no_complex(
            x, y, z,
            self.xx, self.yy,
            self.C1, self.C2,
            pists
        )


def get_executor(slm: SLM):
    return NumpyExecutor(slm)


ε = np.newaxis


def rs_soa_pupil(x, y, z, xx, yy, C1, C2, pists) -> np.ndarray:
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


def rs_soa_pupil_no_complex(x, y, z, xx, yy, C1, C2, pists) -> np.ndarray:
    # in this implementation I remove complex numbers
    slm_p_phase = C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) + \
                  C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] + \
                  2*np.pi*pists[:,ε]

    avg_field = np.vstack((
        np.mean(np.cos(slm_p_phase), axis=0),
        np.mean(np.sin(slm_p_phase), axis=0)
    ))

    return np.arctan2(avg_field[1], avg_field[0])


def compute_metrics_soa_pupil(x, y, z, xx, yy, C1, C2, phase) -> QualityMetrics:
    avg_spot_field = np.mean(
        np.exp(
            1j * (
                phase[ε,:] -
                C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) +
                C2 * z[:,ε] * (xx**2 + yy**2)[ε,:]
            )
        ),
        axis=1
    )

    intensity = np.abs(avg_spot_field)**2
    imin = np.min(intensity)
    imax = np.max(intensity)

    return QualityMetrics(
        efficiency = np.sum(intensity),
        uniformity = 1 - (imax-imin)/(imax+imin),
        variance   = np.sqrt(np.var(intensity)) / np.mean(intensity)
    )

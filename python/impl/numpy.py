import dependency_manager
np = dependency_manager.dep('numpy')

from slm import SLM, QualityMetrics
from executor import Executor


ε = np.newaxis


class NumpyExecutor(Executor):
    def __init__(self, slm: SLM):
        super().__init__(slm)

    def _convert_from_numpy_to_native(self, x: np.ndarray):
        return x

    def _convert_from_native_to_numpy(self, native):
        return native

    def _rs(self, x, y, z, xx, yy, C1, C2, pists):
        return rs_soa_pupil(
            x, y, z,
            xx, yy,
            C1, C2,
            pists
        )


def get_executor(slm: SLM):
    return NumpyExecutor(slm)


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

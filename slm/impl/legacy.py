from slm.common.units import Unit
from slm.common.slm import SLM, QualityMetrics
from slm.common.executor import Executor
from slm.common.loader import load
np = load('numpy')


class LegacyExecutor(Executor):
    def __init__(self, slm: SLM):
        self.slm = slm

    def _convert_from_numpy_to_native(self, x: np.ndarray):
        return x

    def _convert_from_native_to_numpy(self, native):
        return native

    def _rs(self, x, y, z, pists):
        return rs_soa_gen(
            x, y, z,
            self.slm.focal.convert_to(Unit.MILLIMETERS),
            self.slm.pixel_size.convert_to(Unit.MICROMETERS),
            self.slm.wavelength.convert_to(Unit.MICROMETERS),
            self.slm.resolution,
            pists
        )


def get_executor(slm: SLM):
    return LegacyExecutor(slm)


ε = np.newaxis


def rs_soa_gen(x, y, z, focal_mm, pixel_size_um, wavelength_um, resolution, pists) -> np.ndarray:
    NSPOTS = x.shape[0]
    mesh = np.linspace(-1.0, 1.0, num=resolution)
    xx, yy = np.meshgrid(mesh, mesh)

    pupil_idx = np.where(xx**2 + yy**2 < 1.0)
    xx = xx * pixel_size_um * resolution / 2.0
    yy = yy * pixel_size_um * resolution / 2.0
    slm_p_phase = np.zeros((NSPOTS, pupil_idx[0].shape[0]))

    for i in range(NSPOTS):
        slm_p_phase[i,:] = (2.0*np.pi/((wavelength_um)*(focal_mm*10.0**3)))*(x[i]*xx[pupil_idx] + y[i]*yy[pupil_idx]) + ((np.pi*z[i])/(wavelength_um*(focal_mm*10.0**3)**2))*(xx[pupil_idx]**2 + yy[pupil_idx]**2)

    slm_total_field = np.sum(
        (1.0/NSPOTS) * np.exp(
            1j*(slm_p_phase + 2.0*np.pi*pists[:,ε])
        ),
        axis=0
    )

    return np.angle(slm_total_field)

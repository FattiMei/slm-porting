import dependency_manager
np = dependency_manager.dep('numpy')
torch = dependency_manager.dep('torch')

from slm import SLM, QualityMetrics
from executor import Executor


class TorchCpuExecutor(Executor):
    def __init__(self, slm: SLM):
        super().__init__(slm)

    def _convert_from_numpy_to_native(self, x: np.ndarray):
        return torch.from_numpy(x)

    def _convert_from_native_to_numpy(self, native):
        return native.numpy(force=True)

    def _rs(self, x, y, z, pists):
        return rs_soa_pupil(
            x, y, z,
            self.xx, self.yy,
            self.C1, self.C2,
            pists
        )


def get_executor(slm: SLM):
    return TorchCpuExecutor(slm)


ε = None


# the compilation requires "Python.h" to be present in /usr/include/python3.xx
# for Ubuntu just install the development python package "python3-dev"
@torch.compile
def rs_soa_pupil(x, y, z, xx, yy, C1, C2, pists) -> torch.Tensor:
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

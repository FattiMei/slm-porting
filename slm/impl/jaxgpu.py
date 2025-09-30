from slm.common.loader import load
from slm.impl.jaxcpu import JaxCpuExecutor
np = load('numpy')
jax = load('jax')
jnp = jax.numpy


try:
    available_gpus = jax.devices(backend='gpu')
except RuntimeError:
    raise ImportError


# same interface, but we need to make sure to allocate
# the arrays on the gpu
class JaxGpuExecutor(JaxCpuExecutor):
    def __init__(self, slm: SLM):
        super().__init__(slm)

    def _convert_from_numpy_to_native(self, x: np.ndarray):
        pass

    def _convert_from_native_to_numpy(self, x: np.ndarray):
        pass


def get_executor(slm: SLM):
    return JaxGpuExecutor(slm)

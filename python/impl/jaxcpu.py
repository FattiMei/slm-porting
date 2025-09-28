import dependency_manager
np = dependency_manager.dep('numpy')
jax = dependency_manager.dep('jax')
jnp = jax.numpy

from slm import SLM, QualityMetrics
from executor import Executor
from functools import partial


ε = jnp.newaxis


class JaxCpuExecutor(Executor):
    def __init__(self, slm: SLM):
        super().__init__(slm)

    def _convert_from_numpy_to_native(self, x: np.ndarray):
        return jnp.array(x)

    def _convert_from_native_to_numpy(self, native):
        return np.array(native)

    def _rs(self, x, y, z, xx, yy, C1, C2, pists):
        return rs_soa_pupil(
            x, y, z,
            xx, yy,
            C1, C2,
            pists
        )


def get_executor(slm: SLM):
    return JaxCpuExecutor(slm)


@partial(jax.jit, static_argnames=('C1', 'C2'))
def rs_soa_pupil(x, y, z, xx, yy, C1, C2, pists) -> jax.Array:
    return jnp.angle(
        jnp.mean(
            jnp.exp(
                1j * (
                    C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) +
                    C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] +
                    2*jnp.pi*pists[:,ε]
                )
            ),
            axis=0
        )
    )

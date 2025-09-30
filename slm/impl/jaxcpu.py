from functools import partial
from slm.common.slm import SLM, QualityMetrics
from slm.common.executor import Executor
from slm.common.loader import load

np = load('numpy')
jax = load('jax')
jnp = jax.numpy


class JaxCpuExecutor(Executor):
    def __init__(self, slm: SLM):
        super().__init__(slm)

    def _convert_from_numpy_to_native(self, x: np.ndarray):
        return jnp.array(x)

    def _convert_from_native_to_numpy(self, native):
        return np.array(native)

    def _rs(self, x, y, z, pists):
        return rs_soa_pupil(
            x, y, z,
            self.xx, self.yy,
            self.C1, self.C2,
            pists
        )


def get_executor(slm: SLM):
    return JaxCpuExecutor(slm)


ε = jnp.newaxis


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


@partial(jax.jit, static_argnames=('C1', 'C2'))
def rs_soa_pupil_no_complex(x, y, z, xx, yy, C1, C2, pists) -> jax.Array:
    slm_p_phase = C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) + \
                  C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] + \
                  2*jnp.pi*pists[:,ε]

    avg_field = jnp.vstack((
        jnp.mean(jnp.cos(slm_p_phase), axis=0),
        jnp.mean(jnp.sin(slm_p_phase), axis=0)
    ))

    return jnp.arctan2(avg_field[1], avg_field[0])

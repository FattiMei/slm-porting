import dependency_manager
jax = dependency_manager.dep('jax')
jnp = jax.numpy

from common import QualityMetrics
from functools import partial


ε = jnp.newaxis


@partial(jax.jit, static_argnames=('C1', 'C2'))
def rs_soa_pupil(x, y, z, xx, yy, C1, C2, pists) -> jax.Array:
    return jnp.angle(
        jnp.mean(
            jnp.exp(
                1j * (
                    C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) +
                    C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] +
                    2*jnp.pi*pists
                )
            ),
            axis=0
        )
    )

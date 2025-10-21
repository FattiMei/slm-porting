import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from slmporting.core.types import Algorithm, Backend, Device, DType, Tensor
from slmporting.core.contract import impl
from functools import partial


ε = jnp.newaxis


@impl(Algorithm.RS, Backend.JAX, (Device.CPU, Device.GPU),
      compiler = partial(jax.jit, static_argnames=('C1', 'C2')),
      description = 'same code')
def rs(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
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


@impl(Algorithm.RS, Backend.JAX, (Device.CPU, Device.GPU),
      compiler = partial(jax.jit, static_argnames=('C1', 'C2')),
      description = 'simulate complex numbers with real numbers')
def rs_no_complex(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    slm_p_phase = C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) + \
                  C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] + \
                  2*jnp.pi*pists[:,ε]

    avg_field = jnp.vstack((
        jnp.mean(jnp.cos(slm_p_phase), axis=0),
        jnp.mean(jnp.sin(slm_p_phase), axis=0)
    ))

    return jnp.arctan2(avg_field[1], avg_field[0])


@impl(Algorithm.RS, Backend.JAX, (Device.CPU, Device.GPU),
      compiler = partial(jax.jit, static_argnames=('C1', 'C2')),
      description = 'using `jax.vmap` to remove the slm_p_phase allocation')
def rs_vmap(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    def rs_for_a_single_pupil_point(coord):
        xx, yy = coord[0], coord[1]

        return jnp.angle(
            jnp.mean(
                jnp.exp(
                    1j * (
                        C1 * (x*xx + y*yy) +
                        C2 * z * (xx**2 + yy**2) +
                        2*jnp.pi*pists
                    )
                )
            )
        )

    return jax.vmap(rs_for_a_single_pupil_point)(
        jnp.vstack((xx,yy)).T
    )


IMPLS = [rs, rs_no_complex, rs_vmap]

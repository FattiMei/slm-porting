import numpy as np
from contract import implementation, FloatType

# jax imports are critical to enable float64 computation and multicore processing
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_num_cpu_devices', 8) # TODO: query from the system


ε = jnp.newaxis


def conversion_from_numpy_to_native(x: np.ndarray, dtype: FloatType) -> np.ndarray:
    return jnp.array(
        x,
        dtype = {
            FloatType.fp16: 'float16',
            FloatType.fp32: 'float32',
            FloatType.fp64: 'float64'
        }[dtype]
    )


def conversion_from_native_to_numpy(x: jnp.array) -> np.ndarray:
    return np.array(x, dtype=np.float64)


def compilation_function(fcn):
    return partial(jax.jit, static_argnames=(['C1', 'C2']))


def rs_soa(x, y, z, pists, C1: float, C2: float, pixel_size_um: float, resolution: int) -> jax.Array:
    mesh = jnp.linspace(-1.0, 1.0, num=resolution, dtype=x.dtype)
    xx, yy = jnp.meshgrid(mesh,mesh)

    pupil_idx = jnp.where(xx**2 + yy** 2 < 1.0)
    xx = xx[pupil_idx] * pixel_size_um * resolution / 2.0
    yy = yy[pupil_idx] * pixel_size_um * resolution / 2.0

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


def rs_soa_no_complex(x, y, z, pists, C1: float, C2: float, pixel_size_um: float, resolution: int) -> jax.Array:
    mesh = jnp.linspace(-1.0, 1.0, num=resolution, dtype=x.dtype)
    xx, yy = jnp.meshgrid(mesh,mesh)

    pupil_idx = jnp.where(xx**2 + yy** 2 < 1.0)
    xx = xx[pupil_idx] * pixel_size_um * resolution / 2.0
    yy = yy[pupil_idx] * pixel_size_um * resolution / 2.0

    slm_p_phase = C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) + \
                  C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] + \
                  2*jnp.pi*pists[:,ε]

    avg_field = jnp.vstack((
        jnp.mean(jnp.cos(slm_p_phase), axis=0),
        jnp.mean(jnp.sin(slm_p_phase), axis=0)
    ))

    return jnp.arctan2(avg_field[1], avg_field[0])


# TODO: add the manual loop variant. It's particularly important because the jax compiler may perform very good


IMPLS = [
    implementation(rs_soa           , 'rs', 'jax', conversion_from_numpy_to_native, conversion_from_native_to_numpy, compilation_function, description = 'same exact code of numpy version'),
    implementation(rs_soa_no_complex, 'rs', 'jax', conversion_from_numpy_to_native, conversion_from_native_to_numpy, compilation_function, description = 'same exact code of numpy version'),
]

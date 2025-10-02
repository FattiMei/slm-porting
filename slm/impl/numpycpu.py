import itertools
import numpy as np
from impldecorator import implementation


ε = np.newaxis
identity = lambda x: x


@implementation(
    algorithm_class = 'rs',
    backend_name = 'numpy',
    conversion_from_numpy_to_native = identity,
    conversion_from_native_to_numpy = identity,
    description = 'array programming implementation'
)
def rs_soa(x, y, z, pists, C1: float, C2: float, pixel_size_um: float, resolution: int) -> np.ndarray:
    mesh = np.linspace(-1.0, 1.0, num=resolution, dtype=x.dtype)
    xx, yy = np.meshgrid(mesh,mesh)

    pupil_idx = np.where(xx**2 + yy** 2 < 1.0)
    xx = xx[pupil_idx] * pixel_size_um * resolution / 2.0
    yy = yy[pupil_idx] * pixel_size_um * resolution / 2.0

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


@implementation(
    algorithm_class = 'rs',
    backend_name = 'numpy',
    conversion_from_numpy_to_native = identity,
    conversion_from_native_to_numpy = identity,
    description = 'emulating complex numbers with reals'
)
def rs_soa_no_complex(x, y, z, pists, C1: float, C2: float, pixel_size_um: float, resolution: int) -> np.ndarray:
    # complex numbers are treated as a pair of real numbers
    mesh = np.linspace(-1.0, 1.0, num=resolution, dtype=x.dtype)
    xx, yy = np.meshgrid(mesh,mesh)

    pupil_idx = np.where(xx**2 + yy** 2 < 1.0)
    xx = xx[pupil_idx] * pixel_size_um * resolution / 2.0
    yy = yy[pupil_idx] * pixel_size_um * resolution / 2.0

    slm_p_phase = C1 * (x[:,ε]*xx[ε,:] + y[:,ε]*yy[ε,:]) + \
                  C2 * z[:,ε] * (xx**2 + yy**2)[ε,:] + \
                  2*np.pi*pists[:,ε]

    avg_field = np.vstack((
        np.mean(np.cos(slm_p_phase), axis=0),
        np.mean(np.sin(slm_p_phase), axis=0)
    ))

    return np.arctan2(avg_field[1], avg_field[0])


@implementation(
    algorithm_class = 'rs',
    backend_name = 'numpy',
    conversion_from_numpy_to_native = identity,
    conversion_from_native_to_numpy = identity,
    description = 'avoid nspots*npupils allocation'
)
def rs_soa_manual_loop(x, y, z, pists, C1: float, C2: float, pixel_size_um: float, resolution: int) -> np.ndarray:
    # by writing the loop manually I don't store the `slm_p_phase` variable
    # this way I can work with as many spots as I want without allocating 30GB of data
    #
    # I'm expecting this implementation to be slow because of:
    #   * python loops
    #   * list append
    # I'm curios if compiled versions of this variation can be fast
    mesh = np.linspace(-1.0, 1.0, num=resolution, dtype=x.dtype)
    xx, yy = np.meshgrid(mesh,mesh)

    acc = []

    for i in range(mesh.shape[0]):
        for j in range(mesh.shape[1]):
            xx, yy = mesh[i], mesh[j]

            if xx**2 + yy**2 < 1.0:
                xx = xx * pixel_size_um * resolution / 2.0
                yy = yy * pixel_size_um * resolution / 2.0

                acc.append(np.angle(np.mean(np.exp(
                    1j * (
                        C1 * (x * xx + y * yy) +
                        C2 * z * (xx**2 + yy**2) +
                        2*np.pi*pists
                )))))

    return np.array(acc)


@implementation(
    algorithm_class = 'rs',
    backend_name = 'numpy',
    conversion_from_numpy_to_native = identity,
    conversion_from_native_to_numpy = identity,
    description = 'itertools to improve `rs_soa_manual_loop`'
)
def rs_soa_manual_loop_itertools(x, y, z, pists, C1: float, C2: float, pixel_size_um: float, resolution: int) -> np.ndarray:
    mesh = np.linspace(-1.0, 1.0, num=resolution, dtype=x.dtype)

    pupil_points = filter(
        lambda x,y: x**2 + y**2 < 1.0,
        itertools.product(mesh, mesh)
    )

    acc = []

    for (xx, yy) in pupil_points:
        xx = xx * pixel_size_um * resolution / 2.0
        yy = yy * pixel_size_um * resolution / 2.0

        acc.append(np.angle(np.mean(np.exp(
            1j * (
                C1 * (x * xx + y * yy) +
                C2 * z * (xx**2 + yy**2) +
                2*np.pi*pists
        )))))

    return np.array(acc)


# TODO: add an implementation that performs some mathematical
# transformations and don't compute the product with pixel_size and resolution,
# but put them into C1 and C2

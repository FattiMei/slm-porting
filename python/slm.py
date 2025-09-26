import dependency_manager
np = dependency_manager.dep('numpy')

import units
from units import Length
import unittest
from typing import NamedTuple


def precompute_pupil_data(resolution: int, pixel_size: Length, dtype):
    mesh = np.linspace(-1.0, 1.0, num=resolution, dtype=dtype)

    # we will compute only the contributions of points
    # inside a circular pupil
    xx, yy = np.meshgrid(mesh, mesh)
    pupil_idx = np.where(xx**2 + yy**2 < 1.0)

    # maybe `xx` could be of type `Length`, we have proof that it's possible
    # now `xx` is a list of scalars, we don't have anymore the cartesian product structure
    xx = xx[pupil_idx] * pixel_size.convert_to(units.MICRO) * resolution / 2.0
    yy = yy[pupil_idx] * pixel_size.convert_to(units.MICRO) * resolution / 2.0

    return pupil_idx, xx, yy


class SLM:
    def __init__(
        self,
        focal:      Length,
        pixel_size: Length,
        wavelength: Length,
        resolution: int,
        dtype = np.float64
    ):
        self.focal = focal
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.resolution = resolution

        self.pupil_idx, self.xx, self.yy = precompute_pupil_data(
            resolution,
            pixel_size,
            dtype
        )

        self.C1 = 2*np.pi / (wavelength.convert_to(units.MICRO) * focal.convert_to(units.MICRO))
        self.C2 = np.pi / (wavelength.convert_to(units.MICRO) * focal.convert_to(units.MICRO)**2)

    def get_standard_slm():
        focal      = units.Length(20.0, units.MILLI)
        pixel_size = units.Length(15.0, units.MICRO)
        wavelength = units.Length(488.0, units.NANO)
        resolution = 512

        return SLM(focal, pixel_size, wavelength, resolution, dtype)


class QualityMetrics(NamedTuple):
    efficiency: float
    uniformity: float
    variance:   float


class TestSLM(unittest.TestCase):
    def test_constructor(self):
        focal      = units.Length(20.0, units.MILLI)
        pixel_size = units.Length(15.0, units.MICRO)
        wavelength = units.Length(488.0, units.NANO)
        resolution = 512

        for dtype in [np.float16, np.float32, np.float64, np.float128]:
            with self.subTest(dtype=dtype):
                slm = SLM(focal, pixel_size, wavelength, resolution, dtype)

                self.assertTrue(slm.xx.dtype == dtype)
                self.assertTrue(slm.yy.dtype == dtype)


if __name__ == '__main__':
    unittest.main()

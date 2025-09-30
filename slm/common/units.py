import unittest
import itertools
from enum import Enum

from slm.common.loader import load
np = load('numpy')


class Unit(Enum):
    METERS      = 1.0
    MILLIMETERS = 1e-3
    MICROMETERS = 1e-6
    NANOMETERS  = 1e-9


class Length:
    def __init__(self, mantissa: float, unit: Unit = Unit.METERS):
        self.value = mantissa * unit.value

    def convert_to(self, unit: Unit) -> float:
        return self.value / unit.value


class TestLength(unittest.TestCase):
    def test_conversion_down(self):
        mantissa = 1.0
        l = Length(mantissa)
        millimeters = l.convert_to(Unit.MILLIMETERS)

        self.assertTrue(millimeters == l.value * 1000.0)

    def test_conversion_up(self):
        mantissa = 0.324
        l = Length(mantissa, Unit.MICROMETERS)
        millimeters = l.convert_to(Unit.MILLIMETERS)

        self.assertTrue(abs(millimeters - mantissa / 1000.0) < 1e-8)

    def test_identity(self):
        mantissa = 3.14
        units = [Unit.METERS, Unit.MILLIMETERS, Unit.MICROMETERS, Unit.NANOMETERS]

        for (o1, o2) in itertools.product(units, units):
            l = Length(mantissa, o1)
            m = Length(l.convert_to(o2), o2)
            n = Length(m.convert_to(o1), o1)

            # there may be round off errors so the
            # comparison mustn't be exact
            self.assertTrue(abs(l.value - n.value) < 1e-8)

    def test_numpy_integration(self):
        try:
            l = Length(np.random.random(100))

            self.assertTrue(np.allclose(
                l.value * 1000.0,
                l.convert_to(Unit.MILLIMETERS)
            ))

        except ImportError:
            self.skipTest("numpy is not available")


if __name__ == '__main__':
    unittest.main()

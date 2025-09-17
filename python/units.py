import unittest
import itertools
import numpy as np


ONE = 1
MILLI = -3
MICRO = -6
NANO = -9


class Length:
    '''
    In this implementation, I store the values in floating point,
    not a custom mantissa+exponent representation.
    This means it could lose some precision, fortunately the
    interface is good, so we can always improve it without
    changing other code
    '''
    def __init__(self, mantissa: float, order_of_magnitude: int = 1):
        self.value = mantissa * (10**order_of_magnitude)

    def convert_to(self, order_of_magnitude: int) -> float:
        return self.value * (10**-order_of_magnitude)


class TestLength(unittest.TestCase):
    def test_conversion_down(self):
        mantissa = 1.0
        l = Length(mantissa)
        millimeters = l.convert_to(MILLI)

        self.assertTrue(np.allclose(
            millimeters,
            l.value * 1000.0
        ))

    def test_conversion_up(self):
        mantissa = 0.324
        l = Length(mantissa, MICRO)
        millimeters = l.convert_to(MILLI)

        self.assertTrue(np.allclose(
            millimeters,
            mantissa / 1000.0
        ))

    def test_identity(self):
        mantissa = 3.14
        orders = [ONE, MILLI, MICRO, NANO]

        for (o1, o2) in itertools.product(orders, orders):
            l = Length(mantissa, o1)
            m = Length(l.convert_to(o2), o2)
            n = Length(m.convert_to(o1), o1)

            self.assertTrue(np.allclose(l.value, n.value))


if __name__ == '__main__':
    unittest.main()

import unittest
import itertools
from dependency_manager import DEPS


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

    A nice addition is that it works with numpy arrays out of the box
    '''
    def __init__(self, mantissa, order_of_magnitude: int = 1):
        self.value = mantissa * (10**order_of_magnitude)

    def convert_to(self, order_of_magnitude: int):
        return self.value * (10**-order_of_magnitude)


class TestLength(unittest.TestCase):
    def test_conversion_down(self):
        mantissa = 1.0
        l = Length(mantissa)
        millimeters = l.convert_to(MILLI)

        self.assertTrue(millimeters == l.value * 1000.0)

    def test_conversion_up(self):
        mantissa = 0.324
        l = Length(mantissa, MICRO)
        millimeters = l.convert_to(MILLI)

        self.assertTrue(millimeters == mantissa / 1000.0)

    def test_identity(self):
        mantissa = 3.14
        orders = [ONE, MILLI, MICRO, NANO]

        for (o1, o2) in itertools.product(orders, orders):
            l = Length(mantissa, o1)
            m = Length(l.convert_to(o2), o2)
            n = Length(m.convert_to(o1), o1)

            # there may be round off errors so the
            # comparison mustn't be exact
            self.assertTrue(abs(l.value - n.value) < 1e-8)

    def test_numpy_integration(self):
        if 'numpy' in DEPS:
            np = DEPS['numpy']
            l = Length(np.random.random(100), ONE)

            self.assertTrue(np.allclose(
                l.value * 1000.0,
                l.convert_to(MILLI)
            ))

        else:
            self.skipTest("numpy is not available")


if __name__ == '__main__':
    unittest.main()

import unittest
import itertools
import numpy as np
from enum import Enum


class Power(Enum):
    one   = 1
    milli = 10**(-3)
    micro = 10**(-6)
    nano  = 10**(-9)


class Length:
    '''
    In this implementation, I store the values in floating point,
    not a custom mantissa+exponent representation.
    This means it could lose some precision, fortunately the
    interface is good, so we can always improve it without
    changing other code
    '''
    def __init__(self, mantissa: float, order_of_magnitude: Power = Power.one):
        self.value = mantissa * order_of_magnitude.value

    def convert_to(self, order_of_magnitude: Power) -> float:
        return self.value / order_of_magnitude.value


class TestLength(unittest.TestCase):
    def test_conversion_down_base(self):
        l = Length(1.0)
        millimeters = l.convert_to(Power.milli)

        self.assertTrue(np.allclose(
            millimeters,
            l.value * 1000.0
        ))

    def test_identity(self):
        mantissa = 3.14
        exponents = [Power.one, Power.milli, Power.micro, Power.nano]

        for (e1, e2) in itertools.product(exponents, exponents):
            l = Length(mantissa, e1)
            m = Length(l.convert_to(e2), e2)
            n = Length(m.convert_to(e1), e1)

            self.assertTrue(np.allclose(l.value, n.value))


if __name__ == '__main__':
    unittest.main()

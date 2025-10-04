import numpy as np
import itertools
from slmporting.utils.units import Unit, Length

import pytest


all_units = [Unit.METERS, Unit.MILLIMETERS, Unit.MICROMETERS, Unit.NANOMETERS]


def test_conversion_down():
    mantissa = 1.0
    l = Length(mantissa)
    millimeters = l.convert_to(Unit.MILLIMETERS)

    assert(np.allclose(millimeters, 1000.0 * l.value))


def test_conversion_up():
    mantissa = 0.324
    l = Length(mantissa, Unit.MICROMETERS)
    millimeters = l.convert_to(Unit.MILLIMETERS)

    assert(np.allclose(millimeters, mantissa / 1000.0))


@pytest.mark.parametrize("source", all_units)
@pytest.mark.parametrize("dest", all_units)
def test_identity(source: Unit, dest: Unit):
    mantissa = 3.14

    l = Length(mantissa, source)
    m = Length(l.convert_to(dest), dest)
    n = Length(m.convert_to(source), source)

    assert(np.allclose(l.value, n.value))


def test_numpy_integration():
    l = Length(np.random.random(100))

    assert(np.allclose(
        l.value * 1000.0,
        l.convert_to(Unit.MILLIMETERS)
    ))

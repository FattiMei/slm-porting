import slmporting.impl.numpy as impl_numpy


def test_numpy_impl():
    for implementation in impl_numpy.IMPLS:
        concrete = implementation()

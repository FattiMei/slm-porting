import slmporting.impl.numpy as impl_numpy
import slmporting.impl.torch as impl_torch


def test_numpy_impl():
    for implementation in impl_numpy.IMPLS:
        concrete = implementation()


def test_torch_impl():
    for implementation in impl_torch.IMPLS:
        concrete = implementation()

import slmporting.impl.impl_numpy as impl_numpy
import slmporting.impl.impl_torch as impl_torch
import slmporting.impl.impl_jax   as impl_jax


def test_numpy_impl():
    for implementation in impl_numpy.IMPLS:
        concrete = implementation()


def test_torch_impl():
    for implementation in impl_torch.IMPLS:
        concrete = implementation()


def test_jax_impl():
    for implementation in impl_jax.IMPLS:
        concrete = implementation()

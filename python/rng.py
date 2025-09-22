import copy
import unittest

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import numpy as np


'''
Requirements for a Random Number Generator class:
    * gather multiple rng implementations under one interface (e.g both numpy and jax)
    * test the performance of rng implementations
    * obtain reproducible results in non-deterministic algorithms
the latter can be accomplished by the `clone` method

The critical class is the `JaxWrapperRng` because it makes possible to test jax
implementations (which require their own rng to be compiled efficiently) with the
other ones:
    1. the jax implementation is run with a PRNG key
    2. the same key is used to construct a stateful RNG
    3. the other implementations are tested using clones of this stateful RNG

if the jax and non-jax implementations share the same calls to the RNG, they will
roll the same exact numbers. This behaviour is tested below
'''
class Rng:
    def sample(self, shape, dtype):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError


class NumpyRng(Rng):
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self, shape=1, dtype=np.float64):
        return self.rng.random(shape, dtype=dtype)

    def clone(self):
        return copy.deepcopy(self)


class ConstRng(Rng):
    def __init__(self, constant: float):
        self.constant = constant

    def sample(self, shape, dtype=np.float64):
        return self.constant * np.ones(shape, dtype=dtype)

    def clone(self):
        return ConstRng(self.constant)


class JaxWrapperRng(Rng):
    def __init__(self, key):
        self.key = key

    def sample(self, shape=1, dtype=None):
        result = jax.random.uniform(
            self.key,
            shape=shape,
            dtype=dtype
        )

        # this makes the class stateful
        self.key, _ = jax.random.split(self.key)

        return result

    def clone(self):
        return JaxWrapperRng(self.key)


class TestNumpyRng(unittest.TestCase):
    def test_type_promotion(self):
        rng = NumpyRng(seed=42)

        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                pists = 2.0 * dtype(np.pi) * rng.sample(shape=(1,), dtype=dtype)
                self.assertTrue(pists.dtype == dtype)

    def test_clone(self):
        SHAPE = (100,)

        rng = NumpyRng(seed=42)
        clone = rng.clone()

        self.assertTrue(np.allclose(
            rng.sample(SHAPE),
            clone.sample(SHAPE),
            atol=0.0
        ))

    def test_use_after_clone(self):
        SHAPE = (100,)

        rng = NumpyRng(seed=42)
        _ = rng.sample()
        clone = rng.clone()

        self.assertTrue(np.allclose(
            rng.sample(SHAPE),
            clone.sample(SHAPE),
            atol=0.0
        ))


class TestJaxWrapperRng(unittest.TestCase):
    def test_type_promotion(self):
        rng = JaxWrapperRng(jax.random.key(42))

        for dtype in [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]:
            with self.subTest(dtype=dtype):
                pists = 2.0 * jnp.pi * rng.sample(shape=(1,), dtype=dtype)
                self.assertTrue(pists.dtype == dtype)

    def test_statefulness(self):
        SHAPE = (100,)
        rng = JaxWrapperRng(jax.random.key(42))

        first = rng.sample(SHAPE)
        second = rng.sample(SHAPE)

        self.assertFalse(jnp.allclose(first, second, atol=0.0))

    def test_clone(self):
        SHAPE = (100,)

        rng = JaxWrapperRng(jax.random.key(42))
        clone = rng.clone()

        self.assertTrue(jnp.allclose(
            rng.sample(SHAPE),
            clone.sample(SHAPE),
            atol=0.0
        ))

    def test_use_after_clone(self):
        SHAPE = (100,)

        rng = JaxWrapperRng(jax.random.key(42))
        _ = rng.sample()
        clone = rng.clone()

        self.assertTrue(jnp.allclose(
            rng.sample(SHAPE),
            clone.sample(SHAPE),
            atol=0.0
        ))


if __name__ == '__main__':
    unittest.main()

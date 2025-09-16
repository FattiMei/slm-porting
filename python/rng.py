import copy
import unittest
import numpy as np


'''
Requirements for a Random Number Generator class:
    * gather multiple rng implementations under one interface (e.g both numpy and jax)
    * test the performance of rng implementations
    * obtain reproducible results in non-deterministic algorithms

the latter can be accomplished by cloning an RNG or by supplying a constant number
'''
class Rng:
    def __init__(self, seed: int):
        pass

    def sample(self, size: int):
        pass


class NumpyRng(Rng):
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def sample(self, size: int, dtype=np.float64):
        return self.rng.random(size, dtype=dtype)


class ConstRng(Rng):
    def __init__(self, constant: float):
        self.constant = constant

    def sample(self, size: int, dtype=np.float64):
        return self.constant * np.ones(size, dtype=dtype)


class TestRng(unittest.TestCase):
    def test_numpy(self):
        rng = NumpyRng(42)

        for size in [1, 2, 10, 100]:
            x = rng.sample(size)
            self.assertTrue(x.shape == (size,))

    # this is the example for cloning a rng
    def test_numpy_cloning(self):
        rng = NumpyRng(69)
        _ = rng.sample(100)
        clone = copy.deepcopy(rng)

        size = 100
        self.assertTrue(np.all(rng.sample(size) == clone.sample(size)))

    def test_constant_type(self):
        rng = ConstRng(constant=3.14)
        size = 100

        for dtype in [np.float16, np.float32, np.float64, np.float128]:
            x = rng.sample(size, dtype)

            self.assertTrue(x.shape == (size,))
            self.assertTrue(x.dtype == dtype)


if __name__ == '__main__':
    unittest.main()

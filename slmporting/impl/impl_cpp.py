import torch
from slmporting.core.types import Algorithm, Backend, Device, DType, Tensor
from slmporting.core.contract import impl

try:
    from slmporting import config
except ImportError:
    print("[ERROR]: can't find config file. Run a CMake build please")
    exit(1)

# this is a simple solution to make the c++ sources path
# visible from this file
#
# otherwise when this file is included from someone/something else
# the path to search the sources is the one of the includee
torch.ops.load_library(config.meilib_path)


@impl(Algorithm.RS, Backend.CPP, [Device.CPU], description = 'original cpp implementation, with openmp')
def rs(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    return torch.ops.meilib.rs(x, y, z, pists, xx, yy, C1, C2)


@impl(Algorithm.RS, Backend.CPP, [Device.CPU], description = 'uses std::experimental::simd')
def rs_simd(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    return torch.ops.meilib.rs_simd(x, y, z, pists, xx, yy, C1, C2)


@impl(Algorithm.RS, Backend.CPP, [Device.CPU], description = 'manual unrolling')
def rs_simulated_simd(x: Tensor, y: Tensor, z: Tensor, pists: Tensor, xx: Tensor, yy: Tensor, C1: float, C2: float):
    return torch.ops.meilib.rs_simulated_simd(x, y, z, pists, xx, yy, C1, C2)


IMPLS = [rs, rs_simd, rs_simulated_simd]

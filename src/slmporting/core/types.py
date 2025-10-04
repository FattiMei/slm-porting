from enum import Enum
from typing import NamedTuple


class Backend(Enum):
    NUMPY = 0
    JAX   = 1
    TORCH = 2


class Locality(Enum):
    CPU = 0
    GPU = 1


class DType(Enum):
    fp8   = 1
    fp16  = 2
    fp32  = 4
    fp64  = 8


class Algorithm(Enum):
    RS    = 0
    GS    = 1
    WGS   = 2
    CSGS  = 3
    WCSGS = 4


class ProfileInfo(NamedTuple):
    transfer_time: float
    compute_time: float

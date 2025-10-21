import os
from torch.utils.cpp_extension import load
import torch


# this is a simple solution to make the c++ sources path
# visible from this file
#
# otherwise when this file is included from someone/something else
# the path to search the sources is the one of the includee
src_dir = os.path.join(os.path.dirname(__file__), "..", "cpp")
lib = load(
    name="liboptimized",
    sources=[os.path.join(src_dir, "optimized.cpp")],
    extra_cflags=["-O3", "-march=native", "-ftree-vectorize", "-ffast-math", "-fopenmp"],
    verbose=True
)

x = torch.arange(6, dtype=torch.float32)
lib.scale_inplace(x, 2.0)
print("scaled x:", x)

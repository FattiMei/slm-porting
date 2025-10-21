import torch

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

x = torch.arange(6, dtype=torch.float32)
torch.ops.meilib.scale_inplace(x, 2.0)
print("scaled x:", x)

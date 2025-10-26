import numpy as np
import torch

from slmporting.core.types import DType
from slmporting.core.array import CachedArray
from slmporting.utils.units import Length, Unit


class SLM:
    def __init__(
        self,
        focal:      Length,
        pixel_size: Length,
        wavelength: Length,
        resolution: int,
        dtype: DType = DType.fp64
    ):
        self.focal = focal
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.resolution = resolution

        self.C1 = 2*np.pi / (wavelength.convert_to(Unit.MICROMETERS) * focal.convert_to(Unit.MICROMETERS))
        self.C2 =   np.pi / (wavelength.convert_to(Unit.MICROMETERS) * focal.convert_to(Unit.MICROMETERS)**2)

        mesh = torch.linspace(-1.0, 1.0, resolution)
        xx, yy = torch.meshgrid(mesh, mesh)
        mask = xx**2 + yy**2 < 1.0

        self.pupil_idx = torch.where(mask)

        # these arrays need to be aligned so that jax won't complain
        # if I use torch, they are automatically aligned, so no further
        # processing is required
        self.xx = CachedArray(xx[mask] * pixel_size.convert_to(Unit.MICROMETERS) * resolution / 2.0)
        self.yy = CachedArray(yy[mask] * pixel_size.convert_to(Unit.MICROMETERS) * resolution / 2.0)

    def get_standard_slm():
        focal      = Length(20.0, Unit.MILLIMETERS)
        pixel_size = Length(15.0, Unit.MICROMETERS)
        wavelength = Length(488.0, Unit.NANOMETERS)
        resolution = 512

        return SLM(focal, pixel_size, wavelength, resolution)

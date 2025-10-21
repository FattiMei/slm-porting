import jax
from slmporting.core.types import Device


def get_available_devices() -> list[Device]:
    res = []

    try:
        if len(jax.devices(backend='cpu')) > 0:
            res.append(Device.CPU)
    except RuntimeError:
        # this should never happen
        assert(False)

    try:
        if len(jax.devices(backend='gpu')) > 0:
            res.append(Device.GPU)
    except RuntimeError:
        pass

    return res

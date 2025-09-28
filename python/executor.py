import dependency_manager
np = dependency_manager.dep('numpy')

from abc import ABC, abstractmethod
from slm import SLM


class Executor(ABC):
    def __init__(self, slm: SLM):
        self.C1 = slm.C1
        self.C2 = slm.C2
        self.pupil_idx = slm.pupil_idx
        self._load_pupil_data(slm.xx, slm.yy)

    def _load_pupil_data(self, xx: np.ndarray, yy: np.ndarray):
        self.xx = self._convert_from_numpy_to_native(xx)
        self.yy = self._convert_from_numpy_to_native(yy)

    @abstractmethod
    ''' this can be used to select the float dtype'''
    def _convert_from_numpy_to_native(self, x: np.ndarray):
        pass

    @abstractmethod
    def _convert_from_native_to_numpy(self, native):
        pass

    # all the algorithms implemetations, maybe I could do them with some metaprogramming
    @abstractmethod
    def _rs(x, y, z, xx, yy, C1, C2, pists):
        pass

    def rs(x, y, z, pists) -> np.ndarray:
        x = _convert_from_numpy_to_native(x)
        y = _convert_from_numpy_to_native(y)
        z = _convert_from_numpy_to_native(z)
        pists = _convert_from_numpy_to_native(pists)

        return self._convert_from_native_to_numpy(
            self._rs(
                x, y, z,
                self.xx, self.yy,
                self.C1, self.C2,
                pists
            )
        )

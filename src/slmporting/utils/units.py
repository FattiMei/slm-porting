from enum import Enum


class Unit(Enum):
    METERS      = 1.0
    MILLIMETERS = 1e-3
    MICROMETERS = 1e-6
    NANOMETERS  = 1e-9


class Length:
    def __init__(self, mantissa: float, unit: Unit = Unit.METERS):
        self.value = mantissa * unit.value

    def convert_to(self, unit: Unit) -> float:
        return self.value / unit.value

from dataclasses import dataclass
from enum import Enum


@dataclass
class TestRunnerConfig:
    """Konfiguracja dla TestRunner - uniwersalna dla wszystkich algorytmów"""
    skip_first_vector_generation: bool = False
    save_results_after_each_test: bool = False


class FloodSide(Enum):
    """Enum dla kierunków flood fill"""
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3


@dataclass
class FloodConfig:
    """Konfiguracja dla flood fill - używana w generowaniu wektorów"""
    left: bool = True
    right: bool = True
    top: bool = True
    bottom: bool = True

    @classmethod
    def from_string(cls, flood_string: str) -> 'FloodConfig':
        """Tworzy FloodConfig ze stringa typu '1111'"""
        if len(flood_string) != 4:
            raise ValueError(
                f"Flood string musi mieć 4 znaki, otrzymano: {flood_string}"
            )

        return cls(
            left=flood_string[0] == '1',
            right=flood_string[1] == '1',
            top=flood_string[2] == '1',
            bottom=flood_string[3] == '1'
        )

    def to_string(self) -> str:
        """Konwertuje do formatu string '1111'"""
        return (
            ('1' if self.left else '0') +
            ('1' if self.right else '0') +
            ('1' if self.top else '0') +
            ('1' if self.bottom else '0')
        )

    def to_human_readable(self) -> str:
        """Konwertuje do formatu czytelnego dla człowieka"""
        sides = []
        if self.left:
            sides.append("left")
        if self.right:
            sides.append("right")
        if self.top:
            sides.append("top")
        if self.bottom:
            sides.append("bottom")

        return ", ".join(sides) + f", {self.to_string()}"

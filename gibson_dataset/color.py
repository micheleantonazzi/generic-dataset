from typing import List


class Color:
    """
    This class define a color using red, green, blue value
    """
    def __init__(self, red: int, green: int, blue: int):
        self._red = red
        self._green = green
        self._blue = blue

    def RGB(self) -> List[int]:
        """
        This methods create a RGB representation of the color

        :return: a list of int, containing the color values in RGB format
        """
        return [self._red, self._green, self._blue]

    def BGR(self) -> List[int]:
        """
        This methods create a BGR representation of the color

        :return: a list of int, containing the color values in BGR format
        :rtype: list[int]
        """
        return [self._blue, self._green, self._red]
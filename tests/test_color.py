from gibson_dataset.utilities.color import Color


def test_color_rgb():
    color = Color(red=1, green=2, blue=3)
    assert color.RGB() == [1, 2, 3]


def test_color_bgr():
    color = Color(red=1, green=2, blue=3)
    assert color.BGR() == [3, 2, 1]
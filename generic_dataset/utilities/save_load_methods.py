"""
This file contains a lot of method to save and load data from disk
"""
from typing import NoReturn

import cv2
import numpy as np


def save_cv2_image(path: str, data: np.ndarray) -> NoReturn:
    cv2.imwrite(path, data)


def load_cv2_image(path: str) -> np.ndarray:
    return cv2.imread(path)
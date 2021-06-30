"""
This file contains a lot of method to save and load data from disk
"""
import gzip
import io
import json
import pickle
import zlib
from typing import NoReturn, Dict

import cv2
import numpy as np


def save_float(path: str, data: float) -> NoReturn:
    with open(path + '.txt', mode='w') as file:
        file.write(str(data))


def load_float(path: str) -> float:
    with open(path + '.txt', mode='r') as file:
        return float(file.readline())


def save_cv2_image_bgr(path: str, data: np.ndarray) -> NoReturn:
    cv2.imwrite(path + '.png', data)


def load_cv2_image_bgr(path: str) -> np.ndarray:
    return cv2.imread(path + '.png')


def load_cv2_image_grayscale(path: str) -> np.ndarray:
    return cv2.imread(path + '.png', cv2.IMREAD_GRAYSCALE)


def save_compressed_dictionary(path: str, data: Dict) -> NoReturn:
    bytes = io.BytesIO()
    pickle.dump(data, bytes)

    compressed = zlib.compress(bytes.getvalue())
    with open(path + '.dat', mode='wb') as file:
        file.write(compressed)


def load_compressed_dictionary(path: str) -> Dict:
    with open(path + '.dat', mode='rb') as file:
        compressed = file.read()

    bytes = zlib.decompress(compressed)
    return pickle.loads(bytes)


def save_compressed_numpy_array(path: str, data: np.ndarray) -> NoReturn:
    with gzip.open(path + '.tar.gz', mode='w') as file:
        np.save(file, arr=data)


def load_compressed_numpy_array(path: str) -> np.ndarray:
    with gzip.open(path + '.tar.gz', mode='r') as file:
        return np.load(file)
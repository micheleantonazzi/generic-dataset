"""
This file contains a lot of method to save and load data from disk
"""
import gzip
import json
from typing import NoReturn, Dict

import cv2
import numpy as np


def save_float(path: str, data: float) -> NoReturn:
    with open(path + '.txt', mode='w') as file:
        file.write(str(data))


def load_float(path: str) -> float:
    with open(path + '.txt', mode='r') as file:
        return float(file.readline())


def save_cv2_image(path: str, data: np.ndarray) -> NoReturn:
    cv2.imwrite(path + '.png', data)


def load_cv2_image(path: str) -> np.ndarray:
    return cv2.imread(path + '.png')


def save_dictionary_compressed(path: str, data: Dict) -> NoReturn:
    with gzip.open(path + '.tar.gz', mode='w') as file:
        file.write(json.dumps(data).encode('utf-8'))


def load_dictionary_compressed(path: str) -> Dict:
    with gzip.open(path + '.tar.gz', mode='r') as file:
        return json.loads(file.read().decode('utf-8'))


def save_compressed_numpy_array(path: str, data: np.ndarray) -> NoReturn:
    with gzip.open(path + '.tar.gz', mode='w') as file:
        np.save(file, arr=data)


def load_compressed_numpy_array(path: str) -> np.ndarray:
    with gzip.open(path + '.tar.gz', mode='r') as file:
        return np.load(file)
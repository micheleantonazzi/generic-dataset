import generic_dataset.utilities.engine_selector as eg
import numpy as np


def test_check_cuda():
    eg.check_cuda_support()


def test_get_engine():
    engine = eg.get_engine(eg.NUMPY)
    assert np == engine

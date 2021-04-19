import numpy as np
from generic_dataset.gibson_dataset import GibsonSample


def test_setters(use_gpu: bool = False):
    gibson_sample = GibsonSample()
    gibson_sample.set_depth_data(value=np.array([])).set_depth_image(value=np.array([])).set_color_image(value=np.array([]))
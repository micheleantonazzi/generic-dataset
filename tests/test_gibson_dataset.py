import numpy as np
from generic_dataset.gibson_dataset import GibsonSample
import tests.test_pipeline as tp


def test_setters_exists(use_gpu: bool = False):
    gibson_sample = GibsonSample()
    gibson_sample.set_depth_data(value=np.array([])).set_depth_image(value=np.array([])).set_color_image(value=np.array([]))

def test_getter_exists(use_gpu: bool = False):
    gibson_sample = GibsonSample()
    gibson_sample.get_depth_data()
    gibson_sample.get_color_image()
    gibson_sample.get_depth_data()

def test_setter_getter_color_image(use_gpu: bool = False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = GibsonSample().set_color_image(color_image)
    assert np.array_equal(color_image, sample.get_color_image())

def test_setter_getter_depth_data(use_gpu: bool = False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = GibsonSample().set_depth_data(depth_data)
    assert np.array_equal(depth_data, sample.get_depth_data())

def test_setter_getter_depth_image(use_gpu: bool = False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = GibsonSample().set_depth_image(depth_image)
    assert np.array_equal(depth_image, sample.get_depth_image())

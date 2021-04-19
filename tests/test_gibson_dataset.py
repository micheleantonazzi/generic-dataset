import numpy as np
import pytest

from generic_dataset.gibson_dataset import GibsonSample, AnotherActivePipelineException
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
    sample.create_pipeline_for_color_image(use_gpu=use_gpu)
    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_color_image(use_gpu=use_gpu)

    with pytest.raises(AnotherActivePipelineException):
        sample.get_color_image()

    with pytest.raises(AnotherActivePipelineException):
        sample.set_color_image(np.array([]))

    try:
        sample.create_pipeline_for_depth_image(use_gpu=use_gpu)
        sample.create_pipeline_for_depth_data(use_gpu=use_gpu)
    except AnotherActivePipelineException:
        assert False

def test_setter_getter_depth_data(use_gpu: bool = False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = GibsonSample().set_depth_data(depth_data)
    assert np.array_equal(depth_data, sample.get_depth_data())
    sample.create_pipeline_for_depth_data(use_gpu=use_gpu)
    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_depth_data(use_gpu=use_gpu)

    with pytest.raises(AnotherActivePipelineException):
        sample.get_depth_data()

    with pytest.raises(AnotherActivePipelineException):
        sample.set_depth_data(np.array([]))

    try:
        sample.create_pipeline_for_depth_image(use_gpu=use_gpu)
        sample.create_pipeline_for_color_image(use_gpu=use_gpu)
    except AnotherActivePipelineException:
        assert False

def test_setter_getter_depth_image(use_gpu: bool = False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = GibsonSample().set_depth_image(depth_image)
    assert np.array_equal(depth_image, sample.get_depth_image())
    sample.create_pipeline_for_depth_image(use_gpu=use_gpu)
    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_depth_image(use_gpu=use_gpu)

    with pytest.raises(AnotherActivePipelineException):
        sample.get_depth_image()

    with pytest.raises(AnotherActivePipelineException):
        sample.set_depth_image(np.array([]))

    try:
        sample.create_pipeline_for_color_image(use_gpu=use_gpu)
        sample.create_pipeline_for_depth_data(use_gpu=use_gpu)
    except AnotherActivePipelineException:
        assert False

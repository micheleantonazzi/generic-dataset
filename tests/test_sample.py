from gibson_dataset.sample import Sample, AnotherActivePipelineException
import pytest
import tests.test_pipeline as tp
import numpy as np


def test_pipeline_color_image(use_gpu=False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = Sample(color_image=color_image, depth_data=depth_data)
    sample.create_pipeline_for_color_image(use_gpu=use_gpu)
    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_color_image(use_gpu=use_gpu)

    try:
        sample.create_pipeline_for_depth_image(use_gpu=use_gpu)
        sample.create_pipeline_for_depth_data(use_gpu=use_gpu)
    except AnotherActivePipelineException:
        assert False


def test_pipeline_depth_image(use_gpu=False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = Sample(color_image=color_image, depth_data=depth_data)
    sample.create_pipeline_for_depth_image(use_gpu=use_gpu)
    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_depth_image(use_gpu=use_gpu)

    try:
        sample.create_pipeline_for_color_image(use_gpu=use_gpu)
        sample.create_pipeline_for_depth_data(use_gpu=use_gpu)
    except AnotherActivePipelineException:
        assert False


def test_pipeline_depth_data(use_gpu=False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = Sample(color_image=color_image, depth_data=depth_data)
    sample.create_pipeline_for_depth_data(use_gpu=use_gpu)
    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_depth_data(use_gpu=use_gpu)

    try:
        sample.create_pipeline_for_depth_image(use_gpu=use_gpu)
        sample.create_pipeline_for_color_image(use_gpu=use_gpu)
    except AnotherActivePipelineException:
        assert False


def test_depth_image_creation(use_gpu=False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = Sample(color_image=color_image, depth_data=depth_data)
    pipeline = sample.create_pipeline_to_generate_depth_image(use_gpu=use_gpu)
    depth = pipeline.run().get_data()
    assert np.array_equal(depth, depth_image)


def test_get_color_image(use_gpu=False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = Sample(color_image=color_image, depth_data=depth_data)
    pipeline = sample.create_pipeline_for_color_image(use_gpu=use_gpu)

    with pytest.raises(AnotherActivePipelineException):
        sample.get_color_image()

    pipeline.run().get_data()

    assert np.array_equal(sample.get_color_image(), color_image)


def test_get_depth_image(use_gpu=False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = Sample(color_image=color_image, depth_data=depth_data)
    pipeline = sample.create_pipeline_to_generate_depth_image(use_gpu=use_gpu)

    with pytest.raises(AnotherActivePipelineException):
        sample.get_depth_image()

    pipeline.run().get_data()

    assert np.array_equal(sample.get_depth_image(), depth_image)


def test_get_depth_data(use_gpu=False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = Sample(color_image=color_image, depth_data=depth_data)
    pipeline = sample.create_pipeline_for_depth_data(use_gpu=use_gpu)

    with pytest.raises(AnotherActivePipelineException):
        sample.get_depth_data()

    pipeline.run().get_data()

    assert np.array_equal(sample.get_depth_data(), depth_data)


def test_get_pipelines_methods(use_gpu=False):
    color_image, depth_image, depth_data = tp.load_depth_sample()
    sample = Sample(color_image=color_image, depth_data=depth_data)
    pipeline_color_image = sample.create_pipeline_for_color_image(use_gpu=use_gpu)
    pipeline_depth_data = sample.create_pipeline_for_depth_data(use_gpu=use_gpu)
    pipeline_depth_image = sample.create_pipeline_to_generate_depth_image(use_gpu=use_gpu)

    with pytest.raises(AnotherActivePipelineException):
        sample.get_color_image()

    with pytest.raises(AnotherActivePipelineException):
        sample.get_depth_data()

    with pytest.raises(AnotherActivePipelineException):
        sample.get_depth_image()

    assert pipeline_depth_data == sample.get_pipeline_depth_data()
    assert pipeline_depth_image == sample.get_pipeline_depth_image()
    assert pipeline_color_image == sample.get_pipeline_color_image()

    pipeline_depth_image.run().get_data()
    pipeline_depth_data.run().get_data()
    pipeline_color_image.run().get_data()

    assert sample.get_pipeline_depth_data() is None
    assert sample.get_pipeline_depth_image() is None
    assert sample.get_pipeline_color_image() is None

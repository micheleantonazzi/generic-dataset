import queue

import numpy as np
import cv2
import os

import pytest

from generic_dataset.data_pipeline import DataPipeline, PipelineConfigurationException, \
    PipelineAlreadyRunException, PipelineNotExecutedException


def load_depth_sample():
    color_image = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/positive_sample/positive_colorimage_0.png', cv2.IMREAD_COLOR)
    depth_image = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/positive_sample/depth_image.png', cv2.IMREAD_GRAYSCALE)
    depth_data = np.load(os.path.dirname(os.path.abspath(__file__)) + '/positive_sample/depth_data.npy')
    return color_image, depth_image, depth_data


def test_pipeline_configuration(use_gpu: bool = False):
    pipeline = DataPipeline()
    with pytest.raises(PipelineConfigurationException):
        pipeline.run(use_gpu=use_gpu)

    pipeline.set_data(np.array([]))
    with pytest.raises(PipelineConfigurationException):
        pipeline.run(use_gpu=use_gpu)

    pipeline.set_end_function(lambda data, engine: (data, engine))

    with pytest.raises(PipelineNotExecutedException):
        pipeline.get_data()

    pipeline.run(use_gpu=use_gpu)

    with pytest.raises(PipelineAlreadyRunException):
        pipeline.set_data(np.array([]))

    with pytest.raises(PipelineAlreadyRunException):
        pipeline.set_end_function(lambda data: data)

    with pytest.raises(PipelineAlreadyRunException):
        pipeline.run(use_gpu=use_gpu)


def test_get_data(use_gpu: bool = False):
    data = DataPipeline().set_data(data=np.array([])).set_end_function(f=lambda data: data).run(use_gpu=use_gpu).get_data()

    assert np.array_equal(data, np.array([]))

    data = DataPipeline().set_data(data=np.array([])).set_end_function(f=lambda data: np.array([1])).run(use_gpu=use_gpu).get_data()
    assert np.array_equal(data, np.array([1]))


def test_pipeline_operation(use_gpu: bool = False):
    pipeline = DataPipeline().set_data(np.array([])).set_end_function(lambda d:d)

    assert len(pipeline.get_operations()) == 0

    pipeline.add_operation(operation=lambda data, engine: (data, engine))

    assert len(pipeline.get_operations())== 1

    pipeline.add_operation(operation=lambda data, engine: (engine.array([0]), engine))

    pipeline2 = DataPipeline().set_data(data=np.array([1])).set_end_function(lambda d:d).set_operations(pipeline.get_operations())

    assert len(pipeline.get_operations()) == len(pipeline2.get_operations())

    assert np.array_equal(pipeline2.run(use_gpu=use_gpu).get_data(), pipeline.run(use_gpu=use_gpu).get_data())

    with pytest.raises(PipelineAlreadyRunException):
        pipeline2.get_operations()

    with pytest.raises(PipelineAlreadyRunException):
        pipeline2.add_operation(lambda d, e: (d, e))

    with pytest.raises(PipelineAlreadyRunException):
        pipeline2.set_operations(queue.Queue())


def test_pipeline_bgr_to_rgb(use_gpu=False):
    image_bgr = np.tile([[2, 1, 0]], (256, 256)).reshape((256, 256, 3))
    rgb_image = np.tile([0, 1, 2], (256, 256)).reshape((256, 256, 3))
    converted = DataPipeline().set_data(data=image_bgr).set_end_function(f=lambda data: data)\
        .add_operation(lambda data, engine: (data[:, :, [2, 1, 0]], engine)).run(use_gpu=use_gpu).get_data()

    assert np.array_equal(converted, rgb_image)


def test_pipeline_rgb_to_bgr(use_gpu=False):
    image_rgb = np.tile([[0, 1, 2]], (256, 256)).reshape((256, 256, 3))
    bgr_image = np.tile([2, 1, 0], (256, 256)).reshape((256, 256, 3))
    converted = DataPipeline().set_data(data=image_rgb).set_end_function(f=lambda data: data)\
        .add_operation(lambda data, engine: (data[:, :, ::-1], engine)).run(use_gpu=use_gpu).get_data()

    assert np.array_equal(converted, bgr_image)


def test_pipeline_around(use_gpu=False):
    array = np.array([1.111, 2.226])
    rounded = np.around(array, 2)
    converted = DataPipeline().set_data(data=array).set_end_function(f=lambda data: data)\
        .add_operation(lambda data, engine: (engine.around(data, decimals=2), engine)).run(use_gpu=use_gpu).get_data()

    assert np.array_equal(rounded, converted)
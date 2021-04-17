import numpy as np
import cv2
import os

from generic_dataset.utilities.data_pipeline import DataPipeline

def load_depth_sample():
    color_image = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/positive_sample/positive_colorimage_0.png', cv2.IMREAD_COLOR)
    depth_image = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/positive_sample/depth_image.png', cv2.IMREAD_GRAYSCALE)
    depth_data = np.load(os.path.dirname(os.path.abspath(__file__)) + '/positive_sample/depth_data.npy')
    return color_image, depth_image, depth_data

def test_pipeline_bgr_to_rgb(use_gpu=False):
    image_bgr = np.tile([[2, 1, 0]], (256, 256)).reshape((256, 256, 3))
    rgb_image = np.tile([0, 1, 2], (256, 256)).reshape((256, 256, 3))
    converted = DataPipeline(data=image_bgr, use_gpu=use_gpu).convert_bgr_to_rgb().run().get_data()

    assert np.array_equal(converted, rgb_image)


def test_pipeline_rgb_to_bgr(use_gpu=False):
    image_rgb = np.tile([[0, 1, 2]], (256, 256)).reshape((256, 256, 3))
    bgr_image = np.tile([2, 1, 0], (256, 256)).reshape((256, 256, 3))
    converted = DataPipeline(data=image_rgb, use_gpu=use_gpu).convert_rgb_to_bgr().run().get_data()

    assert np.array_equal(converted, bgr_image)

def test_pipeline_around(use_gpu=False):
    array = np.array([1.111, 2.226])
    rounded = np.around(array, 2)
    converted = DataPipeline(data=array, use_gpu=use_gpu).around(2).run().get_data()

    assert np.array_equal(rounded, converted)

def test_pipeline_add_operation(use_gpu=False):
    array = np.array([11, 9, 11, 7, 5, 15])
    array[array > 10] = 10
    def f(data):
        data[data > 10] = 10
        return data
    converted = DataPipeline(data=np.array([11, 9, 11, 7, 5, 15]), use_gpu=use_gpu).add_operation(f).run().get_data()

    assert np.array_equal(array, converted)

def test_pipeline_scale_values(use_gpu=False):
    array = np.array([0, 1, 2])
    final = np.array([0, 2, 4])
    converted = DataPipeline(data=array, use_gpu=use_gpu).scale_values_on_new_max(4).run().get_data()

    assert np.array_equal(final, converted)
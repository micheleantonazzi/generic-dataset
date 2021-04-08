import numpy as np
import cv2

from gibson_dataset.utilities.data_pipeline import DataPipeline

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

def test_pipeline_generate_dept_image(use_gpu=False):
    #DataPipeline(data=self._depth_data, use_gpu=use_gpu).add_operation(function=limit_range).add_operation(lambda data: data * (255 / np.nanmax(data)).astype('uint8'))
    assert 1 == 1
import numpy as np

from gibson_dataset.utilities.data_pipeline import DataPipeline


def test_pipeline_bgr_to_rgb():
    image_bgr = np.tile([[2, 1, 0]], (256, 256)).reshape((256, 256, 3))
    rgb_image = np.tile([0, 1, 2], (256, 256)).reshape((256, 256, 3))
    converted = DataPipeline(data=image_bgr, use_gpu=True).convert_bgr_to_rgb().run().obtain_data()

    assert np.array_equal(converted, rgb_image)


def test_pipeline_rgb_to_bgr():
    image_rgb = np.tile([[0, 1, 2]], (256, 256)).reshape((256, 256, 3))
    bgr_image = np.tile([2, 1, 0], (256, 256)).reshape((256, 256, 3))
    converted = DataPipeline(data=image_rgb, use_gpu=True).convert_rgb_to_bgr().run().obtain_data()

    assert np.array_equal(converted, bgr_image)

def test_pipeline_around():
    array = np.array([1.111, 2.226])
    rounded = np.around(array, 2)
    converted = DataPipeline(data=array, use_gpu=True).around(2).run().obtain_data()

    assert np.array_equal(rounded, converted)
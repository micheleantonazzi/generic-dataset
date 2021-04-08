import numpy as np

from gibson_dataset.utilities.data_pipeline import DataPipeline


def test_pipeline_bgr_to_rgb():
    image_bgr = np.tile([[2, 1, 0]], (256, 256)).reshape((256, 256, 3))
    rgb_image = np.tile([0, 1, 2], (256, 256)).reshape((256, 256, 3))
    converted = DataPipeline(data=image_bgr, use_gpu=False).convert_bgr_to_rgb().run()

    assert np.array_equal(converted, rgb_image)


def test_pipeline_rgb_to_bgr():
    image_rgb = np.tile([[0, 1, 2]], (256, 256)).reshape((256, 256, 3))
    bgr_image = np.tile([2, 1, 0], (256, 256)).reshape((256, 256, 3))
    converted = DataPipeline(data=image_rgb, use_gpu=False).convert_rgb_to_bgr().run()

    assert np.array_equal(converted, bgr_image)

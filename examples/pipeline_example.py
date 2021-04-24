from generic_dataset.data_pipeline import DataPipeline
import numpy as np


run_pipeline_on_gpu = False

red_image = np.array([[255, 0, 0] for _ in range(256 * 256)]).reshape((256, 256, 3))

pipeline_rgb_to_grayscale = DataPipeline() \
    .set_data(data=red_image) \
    .set_end_function(f=lambda d: d) \
    .add_operation(lambda data, engine: (engine.mean(data, axis=2), engine))

grayscale_image = pipeline_rgb_to_grayscale.run(use_gpu=run_pipeline_on_gpu).get_data()


pipeline_rgb_to_bgr = DataPipeline() \
    .set_data(data=red_image) \
    .set_end_function(lambda d: d) \
    .add_operation(lambda data, engine: (data[..., [2, 1, 0]], engine))

bgr_image = pipeline_rgb_to_bgr.run(use_gpu=run_pipeline_on_gpu).get_data()
import os

from generic_dataset.dataset_disk_manager import DatasetDiskManager
from examples.generated_sample import GeneratedSample
import numpy as np

rgb_image = np.array([[255, 0, 0] for _ in range(256 * 256)]).reshape((256, 256, 3))
generated_sample = GeneratedSample(is_positive=False).set_rgb_image(value=rgb_image).set_field_3(value=1)

pipeline = generated_sample.create_pipeline_convert_rgb_to_bgr()
bgr_image = pipeline.run(use_gpu=True).get_data()

dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_folder')
database = DatasetDiskManager(dataset_path=dataset_path, folder_name='folder_1', sample_class=GeneratedSample)

database.save_sample(generated_sample, use_thread=False)



import os
import shutil

from examples.generated_sample_regression import GeneratedSampleRegression
from generic_dataset.dataset_disk_manager import DatasetDiskManager
from examples.generated_sample_classification import GeneratedSampleClassification
import numpy as np

dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_folder')
shutil.rmtree(path=dataset_path, ignore_errors=True)

rgb_image = np.array([[255, 0, 0] for _ in range(256 * 256)]).reshape((256, 256, 3))

# Classification
generated_sample_0 = GeneratedSampleClassification(label=0).set_rgb_image(value=rgb_image).set_field_3(value=1)
generated_sample_1 = GeneratedSampleClassification(label=1).set_rgb_image(value=rgb_image).set_field_3(value=1)
generated_sample_2 = GeneratedSampleClassification().set_rgb_image(value=rgb_image).set_field_3(value=1).set_label(2)

generated_sample_0.create_pipeline_convert_rgb_to_bgr().run(use_gpu=False).get_data()
generated_sample_1.create_pipeline_convert_rgb_to_bgr().run(use_gpu=False).get_data()
generated_sample_2.create_pipeline_convert_rgb_to_bgr().run(use_gpu=False).get_data()


database = DatasetDiskManager(dataset_path=dataset_path, folder_name='folder_classification', sample_class=GeneratedSampleClassification, max_tread=8)

# Save samples
database.save_sample(generated_sample_0, use_thread=False)
database.save_sample(generated_sample_1, use_thread=True)
database.save_sample(generated_sample_2, use_thread=False)

# Load samples using relative count (it depends on their label)
for (label, relative_count) in database.get_absolute_samples_information():
    future = database.load_sample_using_relative_count(label=label, relative_count=relative_count, use_thread=True)
    sample = future.result()
    print(sample.get_label())

# Load sample using absolute count
sample = database.load_sample_using_absolute_count(absolute_count=2, use_thread=False)
assert sample.get_label() == 2



###############################
# Regression
###############################
generated_sample_0 = GeneratedSampleRegression(label=.0).set_rgb_image(value=rgb_image).set_field_3(value=1)
generated_sample_1 = GeneratedSampleRegression(label=.1).set_rgb_image(value=rgb_image).set_field_3(value=1)
generated_sample_2 = GeneratedSampleRegression().set_rgb_image(value=rgb_image).set_field_3(value=1).set_label(0.2)

generated_sample_0.create_pipeline_convert_rgb_to_bgr().run(use_gpu=False).get_data()
generated_sample_1.create_pipeline_convert_rgb_to_bgr().run(use_gpu=False).get_data()
generated_sample_2.create_pipeline_convert_rgb_to_bgr().run(use_gpu=False).get_data()

dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_folder')
database = DatasetDiskManager(dataset_path=dataset_path, folder_name='folder_regression', sample_class=GeneratedSampleRegression)

database.save_sample(generated_sample_0, use_thread=False)
database.save_sample(generated_sample_1, use_thread=True)
database.save_sample(generated_sample_2, use_thread=False)

# Load samples using relative count (it depends on their label)
for (label, relative_count) in database.get_absolute_samples_information():
    sample = database.load_sample_using_relative_count(label=label, relative_count=relative_count, use_thread=False)

# Load sample using absolute count
sample = database.load_sample_using_absolute_count(absolute_count=2, use_thread=False)
assert sample.get_label() == 0.2

# The label is ignored in regression problems
sample = database.load_sample_using_relative_count(label=3, relative_count=2, use_thread=False)
assert sample.get_label() == 0.2




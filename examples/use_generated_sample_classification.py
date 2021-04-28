from examples.generated_sample_classification import GeneratedSampleClassification
import numpy as np
from generic_dataset.sample_generator import SampleGenerator
import generic_dataset.utilities.save_load_methods as slm


rgb_image = np.array([[255, 0, 0] for _ in range(256 * 256)]).reshape((256, 256, 3))
generated_sample = GeneratedSampleClassification(label=1).set_rgb_image(value=rgb_image).set_field_3(value=1)

pipeline = generated_sample.create_pipeline_convert_rgb_to_bgr()

# The get_data method automatically sets the pipeline result in bgr_image inside generated_sample
bgr_image = pipeline.run(use_gpu=True).get_data()

assert np.array_equal(bgr_image, generated_sample.get_bgr_image())

with generated_sample as sync_sample:
    print(sync_sample.field_3_is_positive())
    print(sync_sample.get_label())

GeneratedSampleClass = SampleGenerator(name='GeneratedSampleClass', label_set={-1, 1}).add_field('field_1', field_type=int) \
    .add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .generate_sample_class()

generated_sample = GeneratedSampleClass(label=1)
generated_sample.get_field_1()
generated_sample.set_field_2(np.array([]))

pipeline = generated_sample.create_pipeline_for_field_2()

# The pipeline is empty, so its results is the same as the initial value if field2
# the get_data method automatically sets the pipeline results the correspondent field in sample instance
data = pipeline.run(use_gpu=False).get_data()
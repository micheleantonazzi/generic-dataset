from generic_dataset.sample_generator import SampleGenerator, AnotherActivePipelineException
import numpy as np

GibsonSample = SampleGenerator(name='GibsonSample')\
    .add_field(field_name='depth_data', field_type=np.ndarray, add_to_dataset=True)\
    .add_field(field_name='depth_image', field_type=np.ndarray, add_to_dataset=True)\
    .add_field(field_name='color_image', field_type=np.ndarray, add_to_dataset=True)\
    .generate_sample_class()
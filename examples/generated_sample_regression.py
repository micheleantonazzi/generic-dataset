from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.generic_sample import synchronize_on_fields
from generic_dataset.sample_generator import SampleGenerator
import numpy as np
import generic_dataset.utilities.save_load_methods as slm

pipeline_rgb_to_gbr = DataPipeline().add_operation(lambda data, engine: (data[:, :, [2, 1, 0]], engine))


@synchronize_on_fields(field_names={'field_3'}, check_pipeline=False)
def field_3_is_positive(sample) -> bool:
    return sample.get_field_3() > 0


# To model a regression problem, label_set parameter must be empty
GeneratedSampleRegression = SampleGenerator(name='GeneratedSampleRegression', label_set=set()).add_dataset_field(field_name='rgb_image', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_dataset_field(field_name='bgr_image', field_type=np.ndarray, save_function=slm.save_cv2_image, load_function=slm.load_cv2_image) \
    .add_field(field_name='field_3', field_type=int) \
    .add_custom_pipeline(method_name='create_pipeline_convert_rgb_to_bgr', elaborated_field='rgb_image', final_field='bgr_image', pipeline=pipeline_rgb_to_gbr) \
    .add_custom_method(method_name='field_3_is_positive', function=field_3_is_positive) \
    .generate_sample_class()
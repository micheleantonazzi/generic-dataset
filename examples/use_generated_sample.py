from examples.generated_sample import GeneratedSample
import numpy as np

rgb_image = np.array([[255, 0, 0] for _ in range(256 * 256)]).reshape((256, 256, 3))
generated_sample = GeneratedSample(is_positive=False).set_rgb_image(value=rgb_image).set_field_3(value=1)

pipeline = generated_sample.create_pipeline_convert_rgb_to_bgr()
bgr_image = pipeline.run(use_gpu=True).get_data()

assert np.array_equal(bgr_image, generated_sample.get_bgr_image())

print(generated_sample.field_3_is_positive())
print(generated_sample.get_is_positive())
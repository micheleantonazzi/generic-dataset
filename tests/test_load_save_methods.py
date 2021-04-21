import os
import numpy as np
import generic_dataset.utilities.save_load_methods as slm


test_methods_path = os.path.join(os.path.dirname(__file__), 'test_folder_save_load_methods')
if not os.path.exists(test_methods_path):
    os.mkdir(test_methods_path)


def test_cv2_save_load_methods():
    image = np.array([55 for _ in range(256*256*3)]).reshape((256, 256, 3))
    path = os.path.join(test_methods_path, 'image_cv2.png')

    slm.save_cv2_image(path=path, data=image)
    loaded_image = slm.load_cv2_image(path)

    assert np.array_equal(loaded_image, image)


def test_save_load_dictionary():
    d = {'field1': 1.1, 'field2': 'Hi', 'field3': [0, 1, 2]}
    path = os.path.join(test_methods_path, 'dictionary.tar.gz')

    slm.save_dictionary_compressed(path, d)
    d_loaded = slm.load_dictionary_compressed(path)

    assert d_loaded == d


def test_save_numpy_array():
    array = np.asarray([1.1 for _ in range(256*256)])

    path = os.path.join(test_methods_path, 'numpy_array.tar.gz')

    slm.save_compressed_numpy_array(path, array)
    array_loaded = slm.load_compressed_numpy_array(path)
    assert np.array_equal(array, array_loaded)


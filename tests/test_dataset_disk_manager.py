import os
import shutil

import numpy as np
import pytest

from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.dataset_disk_manager import DatasetDiskManager
from generic_dataset.sample_generator import SampleGenerator
import generic_dataset.utilities.save_load_methods as slm


pipeline_field_1_2 = DataPipeline().add_operation(lambda data, engine: (engine.array([1 for i in range(10000)]), engine))
GeneratedSample = SampleGenerator(name='GeneratedSample') \
    .add_dataset_field(field_name='field_1', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_field(field_name='field_3', field_type=np.ndarray) \
    .add_custom_pipeline(method_name='pipeline_field_1_2', elaborated_field='field_1', final_field='field_2', pipeline=pipeline_field_1_2) \
    .generate_sample_class()

path = os.path.join(os.path.dirname(__file__), 'dataset_folder')


def test_constructor():
    with pytest.raises(FileNotFoundError):
        DatasetDiskManager(dataset_path='random_path', folder_name='folder', sample_class=GeneratedSample)

    DatasetDiskManager(dataset_path=path, folder_name='folder', sample_class=GeneratedSample)
    DatasetDiskManager(dataset_path=path, folder_name='folder1', sample_class=GeneratedSample)
    DatasetDiskManager(dataset_path=path, folder_name='folder1', sample_class=GeneratedSample)


def test_count_samples():
    shutil.rmtree(path, ignore_errors=True)
    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder', sample_class=GeneratedSample)

    assert dataset.get_negative_samples_count() == 0
    assert dataset.get_positive_samples_count() == 0


def test_save_fields():
    shutil.rmtree(path, ignore_errors=True)
    GeneratedSample2 = SampleGenerator('GeneratedSample2').generate_sample_class()
    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder', sample_class=GeneratedSample)

    with pytest.raises(TypeError):
        dataset.save_sample(GeneratedSample2(is_positive=False), False)

    sample = GeneratedSample(is_positive=False).set_field_1(np.array([0.1 for _ in range(10000)])).set_field_2(np.array([0.2 for _ in range(10000)]))
    dataset.save_sample(sample, False)

    sample = GeneratedSample(is_positive=True).set_field_1(np.array([1.1 for _ in range(10000)])).set_field_2(np.array([1.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    sample = GeneratedSample(is_positive=False).set_field_1(np.array([2.1 for _ in range(10000)])).set_field_2(np.array([2.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    sample = GeneratedSample(is_positive=True).set_field_1(np.array([3.1 for _ in range(10000)])).set_field_2(np.array([3.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    assert dataset.get_negative_samples_count() == 2
    assert dataset.get_positive_samples_count() == 2

    assert dataset.get_negative_samples_information() == [(0,), (2,)]
    assert dataset.get_positive_samples_information() == [(1,), (3,)]
    assert dataset.get_absolute_samples_information() == [(False, 0), (True, 0), (False, 1), (True, 1)]


def test_sample_information():
    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder', sample_class=GeneratedSample)

    assert dataset.get_negative_samples_information() == [(0,), (2,)]
    assert dataset.get_positive_samples_information() == [(1,), (3,)]
    assert dataset.get_absolute_samples_information() == [(False, 0), (True, 0), (False, 1), (True, 1)]


def test_load_sample():
    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder', sample_class=GeneratedSample)
    thread = False

    for i in range(len(dataset.get_absolute_samples_information())):
        thread = not thread
        sample = dataset.load_sample_using_absolute_count(absolute_count=i, use_thread=thread)
        if thread:
            sample: GeneratedSample = sample.result()
        assert np.array_equal(np.array([float(str(i) + '.1') for _ in range(10000)]), sample.get_field_1())
        assert np.array_equal(np.array([float(str(i) + '.2') for _ in range(10000)]), sample.get_field_2())

    order = [(False, 0), (True, 0), (False, 1), (True, 1)]
    for i, (is_positive, count) in zip(range(len(order)), order):
        thread = not thread
        sample = dataset.load_sample_using_relative_count(is_positive=is_positive, relative_count=count, use_thread=thread)
        if thread:
            sample: GeneratedSample = sample.result()

        assert np.array_equal(np.array([float(str(i) + '.1') for _ in range(10000)]), sample.get_field_1())
        assert np.array_equal(np.array([float(str(i) + '.2') for _ in range(10000)]), sample.get_field_2())


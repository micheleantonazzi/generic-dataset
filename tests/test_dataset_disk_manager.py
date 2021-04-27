import os
import shutil

import numpy as np
import pytest

from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.dataset_disk_manager import DatasetDiskManager
from generic_dataset.sample_generator import SampleGenerator
import generic_dataset.utilities.save_load_methods as slm


pipeline_field_1_2 = DataPipeline().add_operation(lambda data, engine: (engine.array([1 for i in range(10000)]), engine))
GeneratedSampleClassification = SampleGenerator(name='GeneratedSample', label_set={0, 1}) \
    .add_dataset_field(field_name='field_1', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_field(field_name='field_3', field_type=np.ndarray) \
    .add_custom_pipeline(method_name='pipeline_field_1_2', elaborated_field='field_1', final_field='field_2', pipeline=pipeline_field_1_2) \
    .generate_sample_class()

GeneratedSampleRegression = SampleGenerator(name='GeneratedSample', label_set=set()) \
    .add_dataset_field(field_name='field_1', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_field(field_name='field_3', field_type=np.ndarray) \
    .add_custom_pipeline(method_name='pipeline_field_1_2', elaborated_field='field_1', final_field='field_2', pipeline=pipeline_field_1_2) \
    .generate_sample_class()

path = os.path.join(os.path.dirname(__file__), 'dataset_folder')

shutil.rmtree(path, ignore_errors=True)


def test_constructor():
    with pytest.raises(FileNotFoundError):
        DatasetDiskManager(dataset_path='random_path', folder_name='folder', sample_class=GeneratedSampleClassification)

    with pytest.raises(FileNotFoundError):
        DatasetDiskManager(dataset_path='random_path', folder_name='folder', sample_class=GeneratedSampleRegression)

    DatasetDiskManager(dataset_path=path, folder_name='folder_classification', sample_class=GeneratedSampleClassification)
    DatasetDiskManager(dataset_path=path, folder_name='folder_regression', sample_class=GeneratedSampleRegression)


def test_count_samples():
    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder', sample_class=GeneratedSampleClassification)

    assert dataset.get_sample_total_amount(0) == 0
    assert dataset.get_sample_total_amount(1) == 0
    assert len(dataset.get_samples_absolute_count(0)) == 0
    assert len(dataset.get_samples_absolute_count(1)) == 0

    with pytest.raises(KeyError):
        dataset.get_samples_absolute_count(2)
    with pytest.raises(KeyError):
        dataset.get_sample_total_amount(2)

    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder', sample_class=GeneratedSampleRegression)

    assert dataset.get_sample_total_amount(label=0) == 0
    assert dataset.get_sample_total_amount(label=1) == 0
    assert len(dataset.get_samples_absolute_count(label=0)) == 0
    dataset.get_sample_total_amount(2)


def test_save_fields_classification():
    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder_classification', sample_class=GeneratedSampleClassification)

    with pytest.raises(TypeError):
        dataset.save_sample(GeneratedSampleRegression(label=1.1), False)

    sample = GeneratedSampleClassification(label=0).set_field_1(np.array([0.1 for _ in range(10000)])).set_field_2(np.array([0.2 for _ in range(10000)]))
    dataset.save_sample(sample, False)

    sample = GeneratedSampleClassification(label=1).set_field_1(np.array([1.1 for _ in range(10000)])).set_field_2(np.array([1.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    sample = GeneratedSampleClassification(label=0).set_field_1(np.array([2.1 for _ in range(10000)])).set_field_2(np.array([2.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    sample = GeneratedSampleClassification(label=1).set_field_1(np.array([3.1 for _ in range(10000)])).set_field_2(np.array([3.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    sample = GeneratedSampleClassification(label=0).set_field_1(np.array([4.1 for _ in range(10000)])).set_field_2(np.array([4.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    assert dataset.get_sample_total_amount(label=0) == 3
    assert dataset.get_sample_total_amount(label=1) == 2

    assert dataset.get_samples_absolute_count(label=0) == [0, 2, 4]
    assert dataset.get_samples_absolute_count(label=1) == [1, 3]
    assert dataset.get_absolute_samples_information() == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]


def test_save_fields_regression():
    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder_regression', sample_class=GeneratedSampleRegression)

    with pytest.raises(TypeError):
        dataset.save_sample(GeneratedSampleClassification(label=1), False)

    sample = GeneratedSampleRegression(label=.0).set_field_1(np.array([0.1 for _ in range(10000)])).set_field_2(np.array([0.2 for _ in range(10000)]))
    dataset.save_sample(sample, False)

    sample = GeneratedSampleRegression(label=1.0).set_field_1(np.array([1.1 for _ in range(10000)])).set_field_2(np.array([1.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    sample = GeneratedSampleRegression(label=.0).set_field_1(np.array([2.1 for _ in range(10000)])).set_field_2(np.array([2.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    sample = GeneratedSampleRegression(label=1.0).set_field_1(np.array([3.1 for _ in range(10000)])).set_field_2(np.array([3.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    sample = GeneratedSampleRegression(label=.0).set_field_1(np.array([4.1 for _ in range(10000)])).set_field_2(np.array([4.2 for _ in range(10000)]))
    dataset.save_sample(sample, True).result()

    assert dataset.get_sample_total_amount(label=0) == 5
    assert dataset.get_sample_total_amount(label=1) == 5
    dataset.get_sample_total_amount(label=2)

    assert dataset.get_samples_absolute_count(label=0) == [0, 1, 2, 3, 4]
    assert dataset.get_samples_absolute_count(label=1) == [0, 1, 2, 3, 4]
    assert dataset.get_absolute_samples_information() == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


def test_sample_information():
    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder_classification', sample_class=GeneratedSampleClassification)

    assert dataset.get_sample_total_amount(0) == 3
    assert dataset.get_samples_absolute_count(0) == [0, 2, 4]
    assert dataset.get_sample_total_amount(1) == 2
    assert dataset.get_samples_absolute_count(1) == [1, 3]
    with pytest.raises(KeyError):
        dataset.get_samples_absolute_count(2)

    assert dataset.get_absolute_samples_information() == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]

    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder_regression', sample_class=GeneratedSampleRegression)

    assert dataset.get_sample_total_amount(0) == 5
    assert dataset.get_samples_absolute_count(0) == [0, 1, 2, 3, 4]
    assert dataset.get_sample_total_amount(1) == 5
    assert dataset.get_samples_absolute_count(1) == [0, 1, 2, 3, 4]
    dataset.get_samples_absolute_count(2)

    assert dataset.get_absolute_samples_information() == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


def test_load_sample_classification():
    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder_classification', sample_class=GeneratedSampleClassification)
    thread = False

    for i in range(len(dataset.get_absolute_samples_information())):
        thread = not thread
        sample = dataset.load_sample_using_absolute_count(absolute_count=i, use_thread=thread)
        if thread:
            sample: GeneratedSampleClassification = sample.result()
        assert np.array_equal(np.array([float(str(i) + '.1') for _ in range(10000)]), sample.get_field_1())
        assert np.array_equal(np.array([float(str(i) + '.2') for _ in range(10000)]), sample.get_field_2())

    order = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
    for i, (label, count) in zip(range(len(order)), order):
        thread = not thread
        sample = dataset.load_sample_using_relative_count(label=label, relative_count=count, use_thread=thread)
        if thread:
            sample: GeneratedSampleClassification = sample.result()

        assert np.array_equal(np.array([float(str(i) + '.1') for _ in range(10000)]), sample.get_field_1())
        assert np.array_equal(np.array([float(str(i) + '.2') for _ in range(10000)]), sample.get_field_2())


def test_load_sample_regression():
    dataset = DatasetDiskManager(dataset_path=path, folder_name='folder_regression', sample_class=GeneratedSampleRegression)
    thread = False

    for i in range(len(dataset.get_absolute_samples_information())):
        thread = not thread
        sample = dataset.load_sample_using_absolute_count(absolute_count=i, use_thread=thread)
        if thread:
            sample: GeneratedSampleRegression = sample.result()
        assert np.array_equal(np.array([float(str(i) + '.1') for _ in range(10000)]), sample.get_field_1())
        assert np.array_equal(np.array([float(str(i) + '.2') for _ in range(10000)]), sample.get_field_2())

    order = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    for i, (label, count) in zip(range(len(order)), order):
        thread = not thread
        sample = dataset.load_sample_using_relative_count(label=label, relative_count=count, use_thread=thread)
        if thread:
            sample: GeneratedSampleClassification = sample.result()

        assert np.array_equal(np.array([float(str(i) + '.1') for _ in range(10000)]), sample.get_field_1())
        assert np.array_equal(np.array([float(str(i) + '.2') for _ in range(10000)]), sample.get_field_2())


import inspect
import os

import numpy as np
import pytest

from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.generic_sample import FieldHasIncorrectTypeException, AnotherActivePipelineException, \
    FieldIsNotDatasetPart
from generic_dataset.sample_generator import SampleGenerator, FieldDoesNotExistException
import generic_dataset.utilities.save_load_methods as slm


test_save_load_path = os.path.join(os.path.dirname(__file__), 'test_folder_save_load_methods')
if not os.path.exists(test_save_load_path):
    os.mkdir(test_save_load_path)


pipeline_field_1_2 = DataPipeline().add_operation(lambda data, engine: (engine.array([1 for i in range(10000)]), engine))
GeneratedSample = SampleGenerator(name='GeneratedSample', label_set={0, 1})\
    .add_dataset_field(field_name='field_1', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array)\
    .add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array)\
    .add_field(field_name='field_3', field_type=np.ndarray, default_value=np.array([1, 1])) \
    .add_custom_pipeline(method_name='pipeline_field_1_2', elaborated_field='field_1', final_field='field_2', pipeline=pipeline_field_1_2)\
    .generate_sample_class()


def test_setters_exists(use_gpu: bool = False):
    generated_sample = GeneratedSample(label=1)
    generated_sample.set_field_1(value=np.array([])).set_field_2(value=np.array([]))


def test_getter_exists(use_gpu: bool = False):
    generated_sample = GeneratedSample(label=0)
    generated_sample.get_field_1()
    generated_sample.get_field_2()


def test_setter_getter(use_gpu: bool = False):
    sample = GeneratedSample(label=1)
    assert np.array_equal(sample.get_field_3(), np.array([1, 1]))


    sample = GeneratedSample(label=1).set_field_1(np.array([1]))
    assert np.array_equal(np.array([1]), sample.get_field_1())

    with pytest.raises(FieldHasIncorrectTypeException):
        sample.set_field_1(3)

    pipeline = sample.create_pipeline_for_field_1()

    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_field_1()

    with pytest.raises(AnotherActivePipelineException):
        sample.get_field_1()

    with pytest.raises(AnotherActivePipelineException):
        sample.set_field_1(np.array([]))

    try:
        sample.create_pipeline_for_field_2()
    except AnotherActivePipelineException:
        assert False

    res = pipeline.run(use_gpu=use_gpu).get_data()

    assert np.array_equal(res, np.array([1]))
    new_value = sample.create_pipeline_for_field_1().add_operation(lambda d, e: (e.array([2]), e)).run(use_gpu=use_gpu).get_data()
    assert np.array_equal(np.asarray([2]), new_value)
    assert np.array_equal(new_value, sample.get_field_1())


def tests_pipeline(use_gpu: bool = False):
    sample = GeneratedSample(label=0).set_field_1(np.array([1.1111 for i in range(10000)]))
    pipeline = sample.create_pipeline_for_field_1().add_operation(lambda d, e: (e.around(d, 2), e))
    sample.create_pipeline_for_field_2()

    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_field_1()

    with pytest.raises(AnotherActivePipelineException):
        sample.get_field_1()

    with pytest.raises(AnotherActivePipelineException):
        sample.set_field_1(np.array([]))

    ret = pipeline.run(use_gpu).get_data()

    assert np.array_equal(ret, np.array([1.11 for _ in range(10000)]))
    assert np.array_equal(sample.get_field_1(), np.array([1.11 for i in range(10000)]))

    sample.create_pipeline_for_field_1()


def test_custom_pipeline(use_gpu: bool = False):
    sample = GeneratedSample(label=0).set_field_1(np.array([1.1111 for i in range(10000)]))
    pipeline = sample.pipeline_field_1_2()

    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_field_1()

    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_field_2()

    ret = pipeline.run(use_gpu).get_data()

    assert np.array_equal(ret, np.array([1 for i in range(10000)]))
    assert np.array_equal(sample.get_field_1(), np.array([1.1111 for i in range(10000)]))
    assert np.array_equal(sample.get_field_2(), np.array([1 for i in range(10000)]))

    sample.set_field_2(np.array([2 for i in range(10000)])).create_pipeline_for_field_2().run(use_gpu).get_data()
    assert np.array_equal(sample.get_field_2(), np.array([2 for i in range(10000)]))


def test_get_dataset_fields(use_gpu: bool = False):
    generated = GeneratedSample(label=0)

    generated.create_pipeline_for_field_3()

    dataset_fields = GeneratedSample.GET_DATASET_FIELDS()

    assert 'field_1' in dataset_fields
    assert 'field_2' in dataset_fields
    assert not 'field_3' in dataset_fields


def test_save_load_generic_field(use_gpu: bool = False):
    generated = GeneratedSample(label=1).set_field_1(np.array([1]))

    with pytest.raises(FieldDoesNotExistException):
        generated.save_field(field_name='field', path='', file_name='')

    with pytest.raises(FieldIsNotDatasetPart):
        generated.save_field(field_name='field_3', path='', file_name='')

    with pytest.raises(FileNotFoundError):
        generated.save_field(field_name='field_1', path='wrong_path', file_name='field_1')

    generated.create_pipeline_for_field_1()

    with pytest.raises(AnotherActivePipelineException):
        generated.save_field(field_name='field_1', path=test_save_load_path, file_name='field_1')

    generated.get_pipeline_field_1().run(use_gpu).get_data()

    generated.save_field(field_name='field_1', path=test_save_load_path, file_name='field_1')

    # Load
    with pytest.raises(FieldDoesNotExistException):
        generated.load_field(field_name='field', path='', file_name='field_1')

    with pytest.raises(FieldIsNotDatasetPart):
        generated.load_field(field_name='field_3', path='', file_name='field_3')

    with pytest.raises(FileNotFoundError):
        generated.load_field(field_name='field_1', path='', file_name='field_1')

    with pytest.raises(FileNotFoundError):
        generated.load_field(field_name='field_1', path=test_save_load_path + 'a', file_name='field_')

    generated.create_pipeline_for_field_1()

    with pytest.raises(AnotherActivePipelineException):
        generated.load_field(field_name='field_1', path=test_save_load_path, file_name='field_1')

    generated.get_pipeline_field_1().run(use_gpu).get_data()

    generated.set_field_1(np.array([2]))
    generated.load_field(field_name='field_1', path=test_save_load_path, file_name='field_1')

    assert np.array_equal(generated.get_field_1(), np.array([1]))

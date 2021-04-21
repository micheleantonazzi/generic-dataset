import inspect

import numpy as np
import pytest

from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.sample_generator import SampleGenerator, AnotherActivePipelineException


pipeline_field_1_2 = DataPipeline().add_operation(lambda data, engine: (engine.array([1 for i in range(10000)]), engine))
GeneratedSample = SampleGenerator(name='GeneratedSample').add_field(field_name='field_1', field_type=np.ndarray)\
    .add_field(field_name='field_2', field_type=np.ndarray)\
    .add_custom_pipeline(method_name='pipeline_field_1_2', elaborated_field='field_1', final_field='field_2', pipeline=pipeline_field_1_2)\
    .generate_sample_class()


def test_setters_exists(use_gpu: bool = False):
    generated_sample = GeneratedSample()
    generated_sample.set_field_1(value=np.array([])).set_field_2(value=np.array([]))


def test_getter_exists(use_gpu: bool = False):
    generated_sample = GeneratedSample()
    generated_sample.get_field_1()
    generated_sample.get_field_2()


def test_setter_getter(use_gpu: bool = False):
    sample = GeneratedSample().set_field_1(np.array([1]))
    assert np.array_equal(np.array([1]), sample.get_field_1())

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
    sample = GeneratedSample().set_field_1(np.array([1.1111 for i in range(10000)]))
    pipeline = sample.create_pipeline_for_field_1().add_operation(lambda d, e: (e.around(d, 2), e))
    sample.create_pipeline_for_field_2()

    with pytest.raises(AnotherActivePipelineException):
        sample.create_pipeline_for_field_1()

    with pytest.raises(AnotherActivePipelineException):
        sample.get_field_1()

    with pytest.raises(AnotherActivePipelineException):
        sample.set_field_1(np.array([]))

    ret = pipeline.run(use_gpu).get_data()

    assert np.array_equal(ret, np.array([1.11 for i in range(10000)]))
    assert np.array_equal(sample.get_field_1(), np.array([1.11 for i in range(10000)]))

    sample.create_pipeline_for_field_1()


def test_custom_pipeline(use_gpu: bool = False):
    sample = GeneratedSample().set_field_1(np.array([1.1111 for i in range(10000)]))
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

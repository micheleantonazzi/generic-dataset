import numpy as np
import pytest

from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.sample_generator import SampleGenerator, FieldNameAlreadyExistsException, Sample, \
    FieldHasIncorrectTypeException, AnotherActivePipelineException, FieldDoesNotExistException


def test_add_field():
    generator = SampleGenerator(name='Sample')
    generator.add_field('a', np.ndarray)

    with pytest.raises(FieldNameAlreadyExistsException):
        generator.add_field('a', int)

def test_generate_sample_class():
    generator = SampleGenerator(name='Sample')
    GeneratedClass = generator.generate_sample_class()

    assert isinstance(GeneratedClass, type)
    assert isinstance(GeneratedClass(), Sample)

def test_fields_setter_getter():
    generator = SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_field('field2', str)
    GeneratedClass = generator.generate_sample_class()

    generated = GeneratedClass()

    with pytest.raises(FieldHasIncorrectTypeException):
        generated.set_field('y')

    generated.set_field(np.array([0]))
    generated.set_field2('Hi')

    assert generated.get_field2() == 'Hi'

    with pytest.raises(FieldNameAlreadyExistsException):
        SampleGenerator('S').add_field(field_name='f', field_type=str).add_field(field_name='f', field_type=int)


def test_pipeline_methods():
    GeneratedClass = SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_field('field2', np.ndarray).add_field('field3', int).generate_sample_class()

    generated = GeneratedClass().set_field(np.array([2])).set_field2(np.array([1])).set_field3(1)

    with pytest.raises(AttributeError):
        generated.create_pipeline_for_field3()

    pipeline_field = generated.create_pipeline_for_field()

    with pytest.raises(AnotherActivePipelineException):
        generated.create_pipeline_for_field()

    res = pipeline_field.run(False).get_data()
    generated.create_pipeline_for_field()


def test_custome_pipeline(use_gpu: bool = False):
    with pytest.raises(FieldDoesNotExistException):
        SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_field('field2', np.ndarray)\
            .add_custom_pipeline('m', elaborated_field='f', final_field='field2', pipeline=DataPipeline())

    with pytest.raises(FieldDoesNotExistException):
        SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_field('field2', np.ndarray) \
            .add_custom_pipeline('m', elaborated_field='field', final_field='field22', pipeline=DataPipeline())

    with pytest.raises(FieldHasIncorrectTypeException):
        SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_field('field2', int) \
            .add_custom_pipeline('m', elaborated_field='field', final_field='field2', pipeline=DataPipeline())

    with pytest.raises(FieldHasIncorrectTypeException):
        SampleGenerator('Sample').add_field(field_name='field', field_type=int).add_field('field2', np.ndarray) \
            .add_custom_pipeline('m', elaborated_field='field', final_field='field2', pipeline=DataPipeline())

    GeneratedClass = SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_field('field2', np.ndarray) \
        .add_custom_pipeline('m', elaborated_field='field', final_field='field2', pipeline=DataPipeline().add_operation(operation=lambda data, engine: (engine.asarray([2]), engine)))\
        .generate_sample_class()

    generated = GeneratedClass().set_field(np.array([1, 1])).set_field2(np.array([]))
    pipeline = generated.m()

    with pytest.raises(AnotherActivePipelineException):
        generated.m()

    with pytest.raises(AnotherActivePipelineException):
        generated.get_field2()
    with pytest.raises(AnotherActivePipelineException):
        generated.get_field()

    with pytest.raises(AnotherActivePipelineException):
        generated.get_field2()

    res = pipeline.run(use_gpu).get_data()

    assert np.array_equal(generated.get_field(), np.array([1,1]))
    assert np.array_equal(generated.get_field2(), np.array([2]))
    assert np.array_equal(res, generated.get_field2())
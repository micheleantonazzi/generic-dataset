import numpy as np
import pytest

from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.generic_sample import GenericSample, AnotherActivePipelineException, FieldHasIncorrectTypeException
from generic_dataset.sample_generator import SampleGenerator, FieldNameAlreadyExistsException, \
    FieldDoesNotExistException, \
    MethodAlreadyExistsException, synchronize_on_fields


def test_add_field():
    generator = SampleGenerator(name='Sample')
    generator.add_field('a', np.ndarray)

    with pytest.raises(FieldNameAlreadyExistsException):
        generator.add_field('a', int)


def test_generate_sample_class():
    generator = SampleGenerator(name='Sample')
    GeneratedClass = generator.generate_sample_class()

    assert isinstance(GeneratedClass, type)
    assert isinstance(GeneratedClass(is_positive=False), GenericSample)
    assert not GeneratedClass(is_positive=False).is_positive()
    assert GeneratedClass(is_positive=False).set_is_positive(True).is_positive()


def test_fields_setter_getter():
    generator = SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_dataset_field('field2', str, lambda d:d, lambda d:d)
    GeneratedClass = generator.generate_sample_class()

    generated = GeneratedClass(is_positive=False)

    with pytest.raises(FieldHasIncorrectTypeException):
        generated.set_field('y')

    generated.set_field(np.array([0]))
    generated.set_field2('Hi')

    assert generated.get_field2() == 'Hi'

    with pytest.raises(FieldNameAlreadyExistsException):
        SampleGenerator('S').add_field(field_name='f', field_type=str).add_field(field_name='f', field_type=int)

    with pytest.raises(FieldNameAlreadyExistsException):
        SampleGenerator('S').add_dataset_field(field_name='f', field_type=str, save_function=lambda d:d, load_function=lambda d:d).add_field(field_name='f', field_type=int)

    with pytest.raises(FieldNameAlreadyExistsException):
        SampleGenerator('S').add_dataset_field(field_name='f', field_type=str, save_function=lambda d:d, load_function=lambda d:d)\
            .add_dataset_field(field_name='f', field_type=str, save_function=lambda d:d, load_function=lambda d:d)


def test_pipeline_methods():
    GeneratedClass = SampleGenerator('Sample').add_dataset_field(field_name='field', field_type=np.ndarray, save_function=lambda d:d, load_function=lambda d:d)\
        .add_field('field2', np.ndarray).add_field(
        'field3', int).generate_sample_class()

    generated = GeneratedClass(is_positive=False).set_field(np.array([2])).set_field2(np.array([1])).set_field3(1)
    with pytest.raises(AttributeError):
        generated.create_pipeline_for_field3()

    with pytest.raises(AttributeError):
        generated.get_pipeline_field3()

    pipeline_field = generated.create_pipeline_for_field()

    assert pipeline_field == generated.get_pipeline_field()
    assert pipeline_field != generated.create_pipeline_for_field2()

    with pytest.raises(AnotherActivePipelineException):
        generated.create_pipeline_for_field()

    res = pipeline_field.run(False).get_data()
    assert pipeline_field != generated.create_pipeline_for_field()


def test_custom_pipeline(use_gpu: bool = False):
    with pytest.raises(MethodAlreadyExistsException):
        SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_field('field2', np.ndarray) \
            .add_custom_pipeline('m', elaborated_field='field', final_field='field2', pipeline=DataPipeline()) \
            .add_custom_pipeline('m', elaborated_field='field', final_field='field2', pipeline=DataPipeline())

    with pytest.raises(FieldDoesNotExistException):
        SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_field('field2', np.ndarray) \
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

    GeneratedClass = SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_field('field2',
                                                                                                              np.ndarray) \
        .add_custom_pipeline('m', elaborated_field='field', final_field='field2', pipeline=DataPipeline().add_operation(
        operation=lambda data, engine: (engine.asarray([2]), engine))) \
        .generate_sample_class()

    generated = GeneratedClass(is_positive=False).set_field(np.array([1, 1])).set_field2(np.array([]))
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

    assert np.array_equal(generated.get_field(), np.array([1, 1]))
    assert np.array_equal(generated.get_field2(), np.array([2]))
    assert np.array_equal(res, generated.get_field2())


def test_custom_method():
    with pytest.raises(MethodAlreadyExistsException):
        SampleGenerator('Sample').add_custom_method(method_name='m', function=lambda d:d).add_custom_method(method_name='m', function=lambda m:m)

    @synchronize_on_fields({'field', 'field2'}, check_pipeline=True)
    def f(sample: GenericSample, i: int) -> int:
        sample.set_is_positive(False)
        return i + 1

    GeneratedClass = SampleGenerator('Sample').add_field(field_name='field', field_type=np.ndarray).add_field('field2', np.ndarray) \
        .add_custom_method(method_name='custom_method', function=f).generate_sample_class()

    generated = GeneratedClass(is_positive=False).set_field(np.array([1])).set_field2(np.array([]))
    assert generated.custom_method(1) == 2

    generated.create_pipeline_for_field()

    with pytest.raises(AnotherActivePipelineException):
        generated.custom_method(1)

    generated.get_pipeline_field().run(False).get_data()

    assert generated.custom_method(2) == 3

    generated.set_is_positive(True)
    generated.custom_method(2)

    assert not generated.is_positive()
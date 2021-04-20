from functools import wraps
from typing import Dict, Any, Union, Set, TypeVar, Callable
import numpy as np
from threading import Lock

from generic_dataset.data_pipeline import DataPipeline

TCallable = TypeVar('TCallable', bound=Callable[..., Any])


class Sample:
    pass


class FieldNameAlreadyExistsException(Exception):
    """
    This exception is raised when the user tries to add a field that has the same name as other fields
    """
    def __init__(self, field_name: str):
        super(FieldNameAlreadyExistsException, self).__init__('A field called "{0}" already exists and cannot be added.'.format(field_name))


class FieldDoesNotExistsException(Exception):
    """
    This exception is raised when an action refers to a non-existent field
    """
    def __init__(self, field_name: str):
        super(FieldDoesNotExistsException, self).__init__('You cannot use {0} field: it does not exist!!'.format(field_name))


class AnotherActivePipelineException(Exception):
    """
    This exception is raised when an active pipeline already exists for a field
    """
    def __init__(self, field_name: str):
        super(AnotherActivePipelineException, self).__init__('A Pipeline for the field "{0}" already exists, terminate it before using this method'.format(field_name))


class FieldHasIncorrectTypeException(Exception):
    """
    This exception is raised when an action is performed on a field which has an incompatible type
    """
    def __init__(self, field_name: str):
        super(FieldHasIncorrectTypeException, self).__init__('There is a type issue: the {0} field has an incompatible type!'.format(field_name))


class MethodAlreadyExistsException(Exception):
    """
    This exception is raised when a method to add already exists
    """
    def __init__(self, method_name):
        super(MethodAlreadyExistsException, self).__init__('A method called {0} already exists: you cannot add a new one!!'.format(method_name))


def synchronized_on_fields(fields_name: Set[str], check_pipeline: bool) -> Callable[[TCallable], TCallable]:
    """
    This decorator synchronizes class methods with the fields they use.
    All methods that use the same field are synchronized with respect to the same lock.
    In Addition, it can check also the field's pipeline and eventually raises an exception if there exists an active one.
    :raises AnotherActivePipelineException: if check_pipeline parameter is True and there is an active pipeline for the given field
    :param fields_name: the Set containing the fields name ot synchronize
    :param check_pipeline: if True, the field's pipeline is checked and an exception is raised is there is an active pipeline.
    :return: Callable
    """
    def decorator(method: TCallable) -> TCallable:
        @wraps(method)
        def sync_method(sample, *args, **kwargs):
            locks = [sample._locks[field_name] for field_name in fields_name]
            [lock.acquire() for lock in locks]
            if check_pipeline:
                for field_name in fields_name:
                    if field_name in sample._pipelines.keys() and sample._pipelines[field_name] is not None:
                        [lock.release() for lock in locks]
                        raise AnotherActivePipelineException('Be careful, there is another active pipeline for {0}, please terminate it.'.format(field_name))

            method_result = method(sample, *args, **kwargs)
            [lock.release() for lock in locks]
            return method_result
        return sync_method
    return decorator


class SampleGenerator:
    """
    This object generates customized Sample class according to the needs of the programmer.
    Using a sample generator it is possible to adds fields to the final Sample class.
    For each field, it is possible to specify its name, its type and if it is a dataset member
    (in this case, it is mandatory to specify a method to save and load this field from disk).
    SampleGenerator automatically creates the following default method for each field:
    - get_{field_name}: getter
    - set_{field_name}: setter
    - create_pipeline_for_{field_name}: return a DataPipeline to elaborate the correspondent field
    In addition, other methods can be created and associated to a precise property.
    For example, they could be used to generate and return a predefined pipeline or to execute a generic function,
    specified by the programmer.
    The final generated sample class is thread safe, in the sense that every methods associated to the same field are synchronized.
    This means that two different threads cannot simultaneously execute two methods associated with the same field.
    """
    def __init__(self, name: str):
        self._name = name
        self._fields_name: Set[str] = set()
        self._fields_type: Dict[str, type] = {}
        self._fields_dataset: Dict[str, bool] = {}
        self._custom_methods: Dict[str, Dict[str, Callable]] = {}

    def add_field(self, field_name: str, field_type: type, add_to_dataset: bool = True) -> 'SampleGenerator':
        """
        Adds a field with the given name and type. The created field can be considered part of the dataset or not,
        based on the value of add_to_dataset_field. If it is True, this field is saved and loaded from disk.
        For each field, getters and setters are automatically created.
        If field_type is numpy.ndarray, it is also added a method that creates a DataPipeline for this field.
        :raise FieldNameAlreadyExists: if the field name already exists
        :param field_name: the name of the field
        :type field_name: str
        :param field_type: the type of the field
        :type field_type: type
        :param add_to_dataset: if True, the field will be saved and loaded form disk
        :type add_to_dataset: bool
        :return: the SampleGenerator instance
        :rtype: SampleGenerator
        """

        if field_name in self._fields_name:
            raise FieldNameAlreadyExistsException(field_name=field_name)

        self._fields_name.add(field_name)
        self._fields_type[field_name] = field_type
        self._fields_dataset[field_name] = add_to_dataset

        return self

    def add_custom_pipeline(self, method_name: str, elaborated_field: str, final_field: str, pipeline: DataPipeline) -> 'SampleGenerator':
        """
        Creates a method which returns a predefined pipeline to elaborate a precise field. The pipeline results can be assigned to final_field
        This is useful to create once a pipeline frequently used.
        :raise FieldDoesNotExistsException: if the specified fields do not exists (elaborated and final fields)
        :raise FieldHasIncorrectType: if the field is not a numpy.ndarray (a pipeline can be executed only using a numpy.ndarray)
        :raise MethodAlreadyExists: if the method_name already exists
        :param method_name: the name of the method
        :type method_name: str
        :param elaborated_field: the field to elaborate with Pipeline
        :type elaborated_field: str
        :param final_field: the field to associate the pipeline
        :type final_field: str
        :param pipeline:
        :return: SampleGenerator
        """
        pass

    def generate_sample_class(self) -> 'GeneratedSampleClass':
        """
        Generates the sample class.
        :return: the sample class definition
        """
        class MetaSample(type):
            def __new__(cls, name, bases, class_dict):
                class_dict['__init__'] = self._create_constructor()

                # Add setters, getters, pipeline method
                for field in self._fields_name:
                    class_dict['set_' + field] = self._create_setter(field_name=field)
                    class_dict['get_' + field] = self._create_getter(field_name=field)
                    # Add pipeline methods only if field is a numpy.ndarray
                    if self._fields_type[field] == np.ndarray:
                        class_dict['create_pipeline_for_' + field] = self._create_add_pipeline_method(field_name=field)
                return type.__new__(cls, self._name, bases, class_dict)

        class GeneratedSampleClass(Sample, metaclass=MetaSample):
            pass
        return GeneratedSampleClass

    def _create_constructor(self):
        def __init__(sample):
            sample._fields_name: Set[str] = self._fields_name.copy()
            sample._fields_type: Dict[str, type] = self._fields_type.copy()
            sample._fields_value: Dict[str, Any] = {field_name: None for field_name in sample._fields_name}
            # The fields with a pipeline must be numpy.ndarray
            sample._pipelines: Dict[str, Union[DataPipeline, None]] = {field_name: None for field_name in sample._fields_name if sample._fields_type[field_name] == np.ndarray}
            sample._locks: Dict[str, Lock] = {field_name: Lock() for field_name in sample._fields_name}

        return __init__

    def _create_setter(self, field_name: str):
        field_type: type = self._fields_type[field_name]
        class_name = self._name

        @synchronized_on_fields(fields_name={field_name}, check_pipeline=True)
        def f(sample, value: field_type) -> class_name:
            """
            Sets "{0}" parameter.
            If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
            :raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before setting a new value
            :param value: the value to be assigned to {1}
            :type value: {2}
            :return: the {3} object
            :rtype: {4}
            """
            sample._fields_value[field_name] = value
            return sample

        f.__doc__ = f.__doc__.format(field_name, field_name, field_type.__name__, class_name, class_name)

        return f

    def _create_getter(self, field_name: str):
        field_type: type = self._fields_type[field_name]

        @synchronized_on_fields(fields_name={field_name}, check_pipeline=True)
        def f(sample) -> field_type:
            """
            Return "{0}" value.
            If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
            :raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before getting a new value
            :return: the value of {1}
            :rtype: {2}
            """
            return sample._fields_value[field_name]

        f.__doc__ = f.__doc__.format(field_name, field_name, field_type.__name__)

        return f

    def _create_add_pipeline_method(self, field_name: str):

        @synchronized_on_fields(fields_name={field_name}, check_pipeline=True)
        def f(sample) -> DataPipeline:
            """
            Create and return a new pipeline to elaborate {0}.
            The pipeline is correctly configured, the data to elaborate are {1}
            and the end-function assigns the pipeline results to the sample instance.
            If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
            :raises AnotherActivePipelineException: if another pipeline is active
            :return: a new pipeline instance for {2}
            :rtype DataPipeline
            """
            def assign(data: np.ndarray) -> np.ndarray:
                with sample._locks[field_name]:
                    sample._fields_value[field_name] = data
                    sample._pipelines[field_name] = None
                    return data

            sample._pipelines[field_name] = DataPipeline().set_data(sample._fields_value[field_name]).set_end_function(assign)
            return sample._pipelines[field_name]

        f.__doc__ = f.__doc__.format(field_name, field_name, field_name)
        return f
from functools import wraps
from typing import Dict, Any, Union, Type, Set, TypeVar, Callable
import numpy as np
from threading import Lock

from generic_dataset.utilities.data_pipeline import DataPipeline

TCallable = TypeVar('TCallable', bound=Callable[..., Any])

class Sample:
    pass

class FieldNameAlreadyExistsException(Exception):
    """
    This exception is raised when the user tries to add a field that has the same name as other fields
    """
    def __init__(self, field_name: str):
        super(FieldNameAlreadyExistsException, self).__init__('A field called "{0}" already exists and cannot be added.'.format(field_name))

class AnotherActivePipelineException(Exception):
    """
    This exception is raised when an active pipeline already exists for a field
    """
    def __init__(self, field_name: str):
        super(AnotherActivePipelineException, self).__init__('A Pipeline for the field "{0}" already exists, terminate it before using this method'.format(field_name))

def synchronized_on_field(field_name: str, check_pipeline: bool) -> Callable[[TCallable], TCallable]:
    """
    This decorator synchronizes class methods with the field they use.
    All methods that use the same field are synchronized with respect to the same lock.
    In Addition, it can check also the field's pipeline and eventually raises an exception if there exists an active one.
    :raises AnotherActivePipelineException: if check_pipeline parameter is True and there is an active pipeline for the given field
    :param field_name: the name of the field
    :param check_pipeline: if True, the field's pipeline is checked and an exception is raised is there is an active pipeline.
    :return: Callable
    """
    def decorator(method: TCallable) -> TCallable:
        @wraps(method)
        def sync_method(sample, *args, **kwargs):
            lock = sample._locks[field_name]
            with lock:
                if check_pipeline and sample._pipelines[field_name] is not None:
                    raise AnotherActivePipelineException('Be careful, there is another active pipeline for {0}, please terminate it.'.format(field_name))
                return method(sample, *args, **kwargs)
        return sync_method
    return decorator

class SampleGenerator:
    """
    This object generates sample class according to the needs of the programmer.
    """
    def __init__(self, name: str):
        self._name = name
        self._fields_name: Set[str] = set()
        self._fields_type: Dict[str, type] = {}
        self._fields_dataset: Dict[str, bool] = {}

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

    def generate_sample_class(self) -> 'GeneratedSampleClass':
        """
        Generates the sample class.
        :return: the sample class definition
        """
        class MetaSample(type):
            def __new__(cls, name, bases, class_dict):
                class_dict['__init__'] = self._create_constructor()

                # Add setters
                for field in self._fields_name:
                    class_dict['set_' + field] = self._create_setter(field_name=field)
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


        @synchronized_on_field(field_name=field_name, check_pipeline=False)
        def f(sample, value: field_type) -> class_name:
            """
            Sets "{0}" parameter.
            If the field is an numpy.ndarray and it has an active pipeline, the pipeline is terminated (this operation can take a while).
            :param value: the value to be assigned to {1}
            :type value: {2}
            :return: None
            """
            pipeline = sample._pipelines[field_name]
            if field_name in sample._pipelines.keys() and sample._pipelines[field_name] is not None:
                pipeline.run().get_data()
                sample._pipelines[field_name] = None
            sample._fields_value[field_name] = value

            return sample

        f.__doc__ = f.__doc__.format(field_name, field_name, field_type.__module__ + '.' + field_type.__name__)

        return f

    def _create_getter(self, field_name: str):
        pass

import queue
from abc import ABCMeta
from typing import Dict, Any, Union, Set, TypeVar, Callable, NoReturn
import numpy as np
from threading import RLock

from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.generic_sample import GenericSample, \
    FieldHasIncorrectTypeException, FieldIsNotDatasetPart, synchronize_on_fields

T = TypeVar('T')


class FieldNameAlreadyExistsException(Exception):
    """
    This exception is raised when the user tries to add a field that has the same name as other fields
    """
    def __init__(self, field_name: str):
        super(FieldNameAlreadyExistsException, self).__init__('A field called "{0}" already exists and cannot be added.'.format(field_name))


class FieldDoesNotExistException(Exception):
    """
    This exception is raised when an action refers to a non-existent field
    """
    def __init__(self, field_name: str):
        super(FieldDoesNotExistException, self).__init__('You cannot use {0} field: it does not exist!!'.format(field_name))


class MethodAlreadyExistsException(Exception):
    """
    This exception is raised when a method to add already exists
    """
    def __init__(self, method_name):
        super(MethodAlreadyExistsException, self).__init__('A method called {0} already exists: you cannot add a new one!!'.format(method_name))


class SampleGenerator:
    """
    This object generates customized Sample class according to the needs of the programmer.
    Using a sample generator it is possible to adds fields to the final Sample class.
    For each field, it is possible to specify its name, its type and if it is a dataset member
    (in this case, it is mandatory to specify a method to save and load this field from disk).
    SampleGenerator automatically creates the following default method for each field:
    - get_{field_name}: getter
    - set_{field_name}: setter
    - create_pipeline_for_{field_name}: returns a DataPipeline to elaborate the correspondent field
    - get_{field_name}_pipeline: returns the pipeline instance for the specified field.
    "is_positive" is a default field and it is added automatically to the generated sample class: this field must also be passes to the constructor.
    In addition, other methods can be created and associated to a precise property.
    For example, they could be used to generate and return a predefined pipeline or to execute a generic function,
    specified by the programmer.
    The final generated sample class is thread safe: a lock is associated to each filed
    and all the automatic generated methods must acquire all lock related to all fields they use.
    This means that two different threads cannot simultaneously execute two methods that use the same fields.
    Keep in mind that only automatically created method are thread safe, unlike the custom methods added by the programmer.
    To avoid this issue, the user can decorate the custom functions using the synchronized_on_field decorator before passing them to the "add_custom_method" function.
    Sometimes it could be necessary block a sample instance to execute a series of actions as a atomic operation.
    To this, the sample generated class offers two methods to acquire and release all locks associated to all fields (like saving and loading sample from disk).
    In addition, it is possible to use the sample generated class with a context manager (with statement):
    in this case all locks are acquired and then released.
    """
    def __init__(self, name: str):
        self._name = name
        self._fields_name: Set[str] = {'is_positive'}
        self._fields_type: Dict[str, type] = {'is_positive': bool}
        self._fields_dataset: Dict[str, Dict[str, Callable]] = {}
        self._custom_methods: Dict[str, Callable] = {}

    def add_field(self, field_name: str, field_type: type) -> 'SampleGenerator':
        """
        Adds a field with the given name and type.
        The field is not considered as part of the dataset: it is a simple instance property.
        For each field, getter and setter methods are automatically created.
        If field_type is "numpy.ndarray", it is also added a methods to create and get a DataPipeline to elaborate it.
        :raise FieldNameAlreadyExists: if the field name already exists
        :param field_name: the name of the field
        :type field_name: str
        :param field_type: the type of the field
        :type field_type: type
        :return: the SampleGenerator instance
        :rtype: SampleGenerator
        """

        if field_name in self._fields_name:
            raise FieldNameAlreadyExistsException(field_name=field_name)

        self._fields_name.add(field_name)
        self._fields_type[field_name] = field_type

        return self

    def add_dataset_field(self, field_name: str, field_type: type, save_function: Callable[[str, T], NoReturn], load_function: Callable[[str], T]) -> 'SampleGenerator':
        """
        Adds a field with the given name and type. The field is considered a part of the dataset,
        so it is saved and load from disk by DatasetDiskManager.
        For each field, getter and setter methods are automatically created.
        If field_type is "numpy.ndarray", it is also added a methods to create and get a DataPipeline to elaborate it.
        For a dataset field, it is mandatory to set up the functions to save and load it to disk.
        The save function must have the signature: save_function(path:str, data: data_type) -> NoReturn: ...
        The load function, instead, must be like this: load_function(path: str) -> data_type: ...
        :raise FieldNameAlreadyExists: if the field name already exists
        :param field_name: the name of the field
        :type field_name: str
        :param field_type: the type of the field
        :type field_type: type
        :param save_function: the function to save the field to disk
        :type save_function: Callable
        :param load_function: the function to load the field from disk
        :type load_function: Callable
        :return: the SampleGenerator instance
        :rtype: SampleGenerator
        """
        self.add_field(field_name=field_name, field_type=field_type)
        self._fields_dataset[field_name] = {'save_function': save_function, 'load_function': load_function}
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
        if elaborated_field not in self._fields_name:
            raise FieldDoesNotExistException(field_name=elaborated_field)

        if final_field not in self._fields_name:
            raise FieldDoesNotExistException(field_name=final_field)

        if self._fields_type[elaborated_field] != np.ndarray:
            raise FieldHasIncorrectTypeException(elaborated_field)

        if self._fields_type[final_field] != np.ndarray:
            raise FieldHasIncorrectTypeException(field_name=final_field)

        if method_name in self._custom_methods.keys():
            raise MethodAlreadyExistsException(method_name=method_name)

        self._custom_methods[method_name] = self._create_add_pipeline_method(elaborated_field=elaborated_field, final_field=final_field, operations=pipeline.get_operations())
        return self

    def add_custom_method(self, method_name: str, function: Callable) -> 'SampleGenerator':
        """
        Adds to the generated sample class a generic function as a class method.
        Remember that the function's signature must have at least one parameter (self).
        Annotate your customized function with type hints and docstring: these information are reported in the stub file.
        Remember to annotate the function with the synchronize_on_field decorator to ensure the 'thread safe' condition
        and pipeline checking on the fields that the method uses.
        :raise MethodAlreadyExistsException: if the method name already exists
        :param method_name: the name of the method
        :type method_name: str
        :param function: the function to add as class method
        :type function: Callable
        :return: the sample generator instance
        :rtype: SampleGenerator
        """
        if method_name in self._custom_methods.keys():
            raise MethodAlreadyExistsException(method_name=method_name)

        self._custom_methods[method_name] = function
        return self

    def generate_sample_class(self) -> 'GeneratedGSampleClass':
        """
        Generates the sample class.
        :return: the sample class definition
        """
        class MetaSample(ABCMeta):
            def __new__(cls, name, bases, class_dict):
                class_dict['__init__'] = self._create_constructor()

                # Add setters, getters, pipeline method
                for field in self._fields_name:
                    class_dict['set_' + field] = self._create_setter(field_name=field)
                    class_dict['get_' + field] = self._create_getter(field_name=field)
                    # Add pipeline methods only if field is a numpy.ndarray
                    if self._fields_type[field] == np.ndarray:
                        class_dict['create_pipeline_for_' + field] = self._create_add_pipeline_method(elaborated_field=field, final_field=field)
                        class_dict['get_pipeline_' + field] = self._create_get_pipeline(field)

                # Adds custom methods
                for method_name, func in self._custom_methods.items():
                    class_dict[method_name] = func

                # Methods for saving and loading fields
                class_dict['save_field'] = self._create_save_generic_field()
                class_dict['load_field'] = self._create_load_generic_field()

                # Methods for acquiring and releasing all locks
                class_dict['release_all_locks'], class_dict['acquire_all_locks'] = self._create_acquire_all_lock_functions()
                class_dict['__exit__'], class_dict['__enter__'] = self._create_enter_exit_function()

                return ABCMeta.__new__(cls, self._name, bases, class_dict)

        class GeneratedSampleClass(GenericSample, metaclass=MetaSample):
            def get_dataset_fields(sample) -> Set[str]:
                return sample._fields_dataset.keys()

        # Copy the docstrings of the override methods
        GeneratedSampleClass.get_dataset_fields.__doc__ = GenericSample.get_dataset_fields.__doc__
        GeneratedSampleClass.save_field.__doc__ = GenericSample.save_field.__doc__
        GeneratedSampleClass.load_field.__doc__ = GenericSample.load_field.__doc__
        GeneratedSampleClass.acquire_all_locks.__doc__ = GenericSample.acquire_all_locks.__doc__
        GeneratedSampleClass.release_all_locks.__doc__ = GenericSample.release_all_locks.__doc__

        return GeneratedSampleClass

    def _create_constructor(self):
        def __init__(sample, is_positive: bool):
            super(type(sample), sample).__init__()

            sample._fields_name: Set[str] = self._fields_name.copy()
            sample._fields_type: Dict[str, type] = self._fields_type.copy()
            sample._fields_value: Dict[str, Any] = {field_name: None for field_name in sample._fields_name}
            sample._fields_value['is_positive'] = is_positive
            # The fields with a pipeline must be numpy.ndarray
            sample._pipelines: Dict[str, Union[DataPipeline, None]] = {field_name: None for field_name in sample._fields_name if sample._fields_type[field_name] == np.ndarray}
            sample._locks: Dict[str, RLock] = {field_name: RLock() for field_name in sample._fields_name}
            sample._fields_dataset = self._fields_dataset.copy()
            sample._acquire_lock = RLock()

        return __init__

    def _create_setter(self, field_name: str):
        field_type: type = self._fields_type[field_name]
        class_name = self._name

        @synchronize_on_fields(fields_name={field_name}, check_pipeline=True)
        def f(sample, value: field_type) -> class_name:
            """
            Sets "{0}" parameter.
            If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
            :raises FieldHasIncorrectTypeException: if the given value has a wrong type
            :raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before setting a new value
            :param value: the value to be assigned to {1}
            :type value: {2}
            :return: the {3} object
            :rtype: {4}
            """
            if sample._fields_type[field_name] != type(value):
                raise FieldHasIncorrectTypeException(field_name)
            sample._fields_value[field_name] = value
            return sample

        f.__doc__ = f.__doc__.format(field_name, field_name, field_type.__name__, class_name, class_name)

        return f

    def _create_getter(self, field_name: str):
        field_type: type = self._fields_type[field_name]

        @synchronize_on_fields(fields_name={field_name}, check_pipeline=True)
        def f(sample) -> field_type:
            """
            Return "{0}" value.
            If the field is an numpy.ndarray and it has an active pipeline, an exception is raised. Terminate it before get the fields value
            :raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before getting a new value
            :return: the value of {1}
            :rtype: {2}
            """
            return sample._fields_value[field_name]

        f.__doc__ = f.__doc__.format(field_name, field_name, field_type.__name__)

        return f

    def _create_add_pipeline_method(self, elaborated_field: str, final_field: str, operations: queue.Queue = None):
        fields = {elaborated_field, final_field}

        @synchronize_on_fields(fields_name=fields, check_pipeline=True)
        def f(sample) -> DataPipeline:
            """
            Creates and returns a new pipeline to elaborate "{0}".
            The pipeline is correctly configured, the data to elaborate are "{1}"
            and the pipeline results is set to "{2}".
            If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
            :raise AnotherActivePipelineException: if other pipeline of the fields are active
            :return: a new pipeline instance which elaborates "{3}" and writes the result into "{4}"
            :rtype: DataPipeline
            """
            def assign(data: np.ndarray) -> np.ndarray:
                with sample._acquire_lock:
                    [sample._locks[field].acquire() for field in fields]
                sample._fields_value[final_field] = data
                for field in fields:
                    sample._pipelines[field] = None
                [sample._locks[field].release() for field in fields]
                return data

            pipeline_configured = DataPipeline().set_data(sample._fields_value[elaborated_field]).set_end_function(assign)
            if operations !=  None:
                pipeline_configured.set_operations(operations)
            for field in fields:
                sample._pipelines[field] = pipeline_configured

            return pipeline_configured

        f.__doc__ = f.__doc__.format(elaborated_field, elaborated_field, final_field, elaborated_field, final_field)
        return f

    def _create_save_generic_field(self):
        class_name = self._name

        def f(sample, field_name: str, path: str) -> class_name:
            if field_name not in sample._fields_name:
                raise FieldDoesNotExistException(field_name=field_name)

            if field_name not in sample._fields_dataset.keys():
                raise FieldIsNotDatasetPart('You cannot save {0}: it is not a part of the dataset!'.format(field_name))

            @synchronize_on_fields(fields_name={field_name}, check_pipeline=True)
            def wrapped_save_function(sample):
                sample._fields_dataset[field_name]['save_function'](path, sample._fields_value[field_name])

            wrapped_save_function(sample)
            return sample

        return f

    def _create_load_generic_field(self):
        class_name = self._name

        def f(sample, field_name: str, path: str) -> class_name:
            if field_name not in sample._fields_name:
                raise FieldDoesNotExistException(field_name=field_name)

            if field_name not in sample._fields_dataset.keys():
                raise FieldIsNotDatasetPart('You cannot save {0}: it is not a part of the dataset!'.format(field_name))

            @synchronize_on_fields(fields_name={field_name}, check_pipeline=True)
            def wrapped_load_function(sample):
                sample._fields_value[field_name] = sample._fields_dataset[field_name]['load_function'](path)

            wrapped_load_function(sample)
            return sample

        return f

    def _create_get_pipeline(self, field_name: str):

        @synchronize_on_fields(fields_name={field_name}, check_pipeline=False)
        def f(sample) -> DataPipeline:
            """
            Returns the pipeline of {0}. If there isn't an active pipeline, returns None.
            :return: the pipeline if it exists, None otherwise
            :rtype: Union[None, DataPipeline]
            """
            return sample._pipelines[field_name]

        f.__doc__ = f.__doc__.format(field_name)
        return f

    def _create_acquire_all_lock_functions(self):
        class_name = self._name
        def acquire_locks(sample) -> class_name:
            with sample._acquire_lock:
                [lock.acquire() for lock in sample._locks.values()]
            return sample

        def release_locks(sample) -> class_name:
            [lock.release() for lock in sample._locks.values()]
            return sample

        return release_locks, acquire_locks

    def _create_enter_exit_function(self):
        class_name = self._name

        def __enter__(sample) -> class_name:
            return sample.acquire_all_locks()

        def __exit__(sample, exc_type, exc_value, exc_traceback):
            sample.release_all_locks()

        return __exit__, __enter__

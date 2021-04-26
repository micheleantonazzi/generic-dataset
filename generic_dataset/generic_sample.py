from abc import abstractmethod, ABCMeta
from functools import wraps
from typing import Set, Any, TypeVar, Callable


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


class FieldIsNotDatasetPart(Exception):
    """
    This exception is raised when a field is not a part of dataset
    """


TCallable = TypeVar('TCallable', bound=Callable[..., Any])


def synchronize_on_fields(field_names: Set[str], check_pipeline: bool) -> Callable[[TCallable], TCallable]:
    """
    This decorator synchronizes class methods with the fields they use.
    All methods that use the same fields are synchronized with respect to the same locks.
    In Addition, it can check also the field's pipeline and eventually raises an exception if there exists an active one.
    :raises AnotherActivePipelineException: if check_pipeline parameter is True and there is an active pipeline for the given field
    :param field_names: the Set containing the fields name ot synchronize
    :param check_pipeline: if True, the field's pipeline is checked and an exception is raised is there is an active pipeline.
    :return: Callable
    """
    def decorator(method: TCallable) -> TCallable:
        @wraps(method)
        def sync_method(self, *args, **kwargs):
            locks = [self._locks[field_name] for field_name in field_names]
            try:
                with self._acquire_lock:
                    [lock.acquire() for lock in locks]
                if check_pipeline:
                    for field_name in field_names:
                        if field_name in self._pipelines.keys() and self._pipelines[field_name] is not None:
                            raise AnotherActivePipelineException('Be careful, there is another active pipeline for {0}, please terminate it.'.format(field_name))

                return method(self, *args, **kwargs)
            finally:
                [lock.release() for lock in locks]

        return sync_method
    return decorator


class GenericSample(metaclass=ABCMeta):
    """
    This base class represents a generic sample, which can be specialized using SampleGenerator.
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_is_positive(self) -> bool:
        pass

    @abstractmethod
    def set_is_positive(self, is_positive: bool) -> 'GenericSample':
        pass

    @abstractmethod
    def get_dataset_fields(self) -> Set[str]:
        """
        Returns the parameter names that belong to the dataset (those must be saved and load from disk)
        :return: the parameter names set
        :rtype: Set[str]
        """
        pass

    @abstractmethod
    def save_field(self, field_name: str, path: str, file_name: str) -> 'GenericSample':
        """
        Saves the given field to disk, in the given path.
        :raise FieldDoesNotExistException: if field_name do not exist in this sample class
        :raise FieldIsNotDatasetPart: if the field is not a part of dataset
        :raise AnotherActivePipelineException if there is an active pipeline for this field
        :raise FileNotFoundError: if the path does not exist
        :param field_name: the name of the field to save
        :type field_name: str
        :param path: the path where save the field value
        :type path: str
        :param file_name: the name of the file in which to save the field. The file extension is automatically added by the save function, so don't include it in the name
        :type file_name: str
        :returns the sample instance
        """
        pass

    @abstractmethod
    def load_field(self, field_name: str, path: str, file_name: str) -> 'GenericSample':
        """
        Loads the given field from disk, saved in the given path.
        The field value is not returned by this method but it is set to the sample class.
        To retrieve it use the correspondent get method.
        :raise FieldDoesNotExistException: if field_name do not exist in this sample class
        :raise FieldIsNotDatasetPart: if the field is not a part of dataset
        :raise AnotherActivePipelineException if there is an active pipeline for this field
        :raise FileNotFoundError: if the path does not exist
        :param field_name: the name of the field to save
        :type field_name: str
        :param path: the path where loading the field value
        :type path: str
        :param file_name: the name of the file in which the field si saved. The file extension is automatically added by the load function, so don't include it in the name
        :type file_name: str
        :return: the sample instance
        """
        pass

    @abstractmethod
    def acquire_all_locks(self) -> 'GenericSample':
        """
        Acquires all locks related to all fields of the sample instance
        :return: GenericSample instance
        """
        pass

    @abstractmethod
    def release_all_locks(self) -> 'GenericSample':
        """
        Releases all locks related to all fields of the sample instance
        :return: GenericSample instance
        """
        pass
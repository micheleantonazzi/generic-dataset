from abc import abstractmethod, ABCMeta
from typing import Set


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


class GenericSample(metaclass=ABCMeta):
    """
    This base class represents a generic sample, which can be specialized using SampleGenerator.
    """
    def __init__(self, is_positive: bool):
        self._is_positive = is_positive

    def is_positive(self) -> bool:
        return self._is_positive

    def set_is_positive(self, is_positive: bool) -> 'GenericSample':
        self._is_positive = is_positive
        return self

    @abstractmethod
    def get_dataset_fields(self) -> Set[str]:
        """
        Returns the parameter names that belong to the dataset (those must be saved and load from disk)
        :return: the parameter names set
        :rtype: Set[str]
        """
        pass

    @abstractmethod
    def save_field(self, field_name: str, path: str) -> 'GenericSample':
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
        :returns the sample instance
        """
        pass

    @abstractmethod
    def load_field(self, field_name: str, path: str) -> 'GenericSample':
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
        :return: the sample instance
        """
        pass

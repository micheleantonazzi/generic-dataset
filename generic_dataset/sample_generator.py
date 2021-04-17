from typing import Dict, Any, Union, Type

class Sample:
    pass

class FieldNameAlreadyExistsException(Exception):
    """
    This exception is raised when the user tries to add a field that has the same name as other fields
    """
    def __init__(self, field_name: str):
        super(FieldNameAlreadyExistsException, self).__init__('A field called "{0}" already exists and cannot be added.'.format(field_name))

class SampleGenerator:
    """
    This object generates sample class according to the needs of the programmer.
    """
    def __init__(self, name: str):
        self._name = name
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

        if field_name in self._fields_type.keys():
            raise FieldNameAlreadyExistsException(field_name=field_name)

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
                return type.__new__(cls, name, bases, class_dict)

        class GeneratedSampleClass(Sample, metaclass=MetaSample):
            pass
        return GeneratedSampleClass

    def _create_constructor(self):
        def __init__(sample):
            sample._fields: Dict[str, Any] = {field_name: None for field_name in self._fields_type.keys()}

        return __init__
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


class GenericSample:
    """
    This base class represents a generic sample, which can be specialized using SampleGenerator.
    """
    pass

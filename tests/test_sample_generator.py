import numpy as np
import pytest

from generic_dataset.sample_generator import SampleGenerator, FieldNameAlreadyExistsException, Sample


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
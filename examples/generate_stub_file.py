import os
from stub_generator.stub_generator import StubGenerator

StubGenerator(os.path.dirname(__file__) + '/generated_sample_classification.py', ['GeneratedSampleClassification']).generate_stubs().write_to_file()
StubGenerator(os.path.dirname(__file__) + '/generated_sample_regression.py', ['GeneratedSampleRegression']).generate_stubs().write_to_file()

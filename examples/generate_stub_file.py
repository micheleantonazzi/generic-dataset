import os
from stub_generator.stub_generator import StubGenerator

StubGenerator(os.path.dirname(__file__) + '/generated_sample.py', ['GeneratedSample']).generate_stubs().write_to_file()

import os
import re
import subprocess
import sys
import platform

# To use a consistent encoding
from codecs import open as copen

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with copen(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with copen(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("generic_dataset", "__version__.py")

test_deps = [
    "pytest",
    "pytest-cov",
    "opencv-python"
]

extras = {
    'test': test_deps,
}

def get_cuda_version():

    res = subprocess.check_output('nvcc --version'.split(' ')).decode()
    if bool(res.rstrip()):
        regex = r'release (\S+),'
        match = re.search(regex, res)
        if match:
            return str(match.group(1)).replace('.', '')

    return ''

# On python 3.6 and aarch64, install cupy (requires a long build)
if sys.version_info[0] == 3 and sys.version_info[1] == 6 and platform.machine() == 'aarch64':
    cuda_version = ''
else:
    try:
        cuda_version = get_cuda_version()
        if not cuda_version == '':
            cuda_version = 'cupy-cuda' + cuda_version
    except:
        cuda_version = ''

setup(
    name='generic-dataset',
    version=__version__,
    description="Gibson dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/micheleantonazzi/generic-dataset",
    author="Michele Antonazzi",
    author_email="micheleantonazzi@gmail.com",
    # Choose your license
    license='MIT',
    include_package_data=True,
    classifiers=[
        'Environment :: GPU :: NVIDIA CUDA',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    tests_require=test_deps,
    # Add here the package dependencies
    install_requires=[
        'opencv-python<4.7; python_version == "3.6"',
        'opencv-python; python_version > "3.6"',
        'numpy',
        'pandas',
        cuda_version,
        'termcolor',
        'stub-generator'
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    test_deps=test_deps,
    extras_require=extras,
)
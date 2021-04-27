# Generic dataset
![](https://github.com/micheleantonazzi/generic-dataset/workflows/Build/badge.svg?branch=main)[
![pypi](https://img.shields.io/pypi/v/generic-dataset.svg)](https://pypi.org/project/generic-dataset/)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=coverage)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)



[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=alert_status)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=sqale_rating)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=reliability_rating)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=security_rating)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=vulnerabilities)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)

This library is designed to support machine learning and data science programmers. In fact, they work with a datasets, but each dataset is different from the others. 'generic-dataset' is a **configurable framework** which generates automatically the code and the necessary classes to manage a dataset of any kind. This is possible using the **metaprogramming paradigm**. In addition, it also offers useful utility to manipulate *numpy arrays*. This utility build a pipeline (a series of operations to modify an array) which can also be easily run on **GPU**. For this reason, this library is particularly suitable for image datasets.

## Installation

'generic-dataset' can be easily installed using pip:

```bash
pip install generic-dataset
```

Otherwise, you can clone this repository in your machine and and then install it:

```bash
git clone https://github.com/micheleantonazzi/generic-dataset
cd generic-dataset/
pip install -e .
```

 ### GPU support

This library can accelerate the operations performed over numpy array using *Nvidia GPU*. This is possible using [CuPy](https://cupy.dev/) framework, which offers an interface highly compatible than NumPy, but all its functionalities are executed on GPU.

**NB:** you can use this library even without a GPU: if you try to use it, an exception is raised.

To enable the GPU support, please install and configure CUDA Toolkit before install this package: this will automatically install CuPy during the installation of *generic-dataset*. Otherwise, you can configure the CUDA environment at a later time and then install CuPy using its [installation guide](https://docs.cupy.dev/en/stable/install.html#installing-cupy).

## Library structure
This library is composed by three main entities: 

* **SampleGenerator:** it is a configurable class which generates **Sample** class according to the programmers needs. The *Sample* class represents a dataset's sample: it stored the fields value, the operation to manipulate them and its state (positive or negative). *Sample* class it and  is not a-priori defined, but it is composed buy *SampleGenerator* using the *metaprogramming paradigm*.
* **DatasetDiskManager:** this class is responsible for storing the dataset. It can take any type of generated sample class and store or load it from disk.
* **DataPipeline:** this entity defines an elaboration pipeline to manipulate numpy arrays. A pipeline can be executed in CPU or GPU.

### SampleGenerator

*SampleGenerator* builds *Sample* class according to the programmer needs. Using this class, you can compose your customized sample of your dataset. Using *SampleGenerator*, the programmer can add fields and custom methods to the generated sample class. The sample class, built by *SampleGenerator*, has the following characteristics:

* It is a sub-type of *GenericSample*. This abstract class defines some abstract methods which are implemented by the generated sample class
* It provides a field called *is_positive* and the relatives getter and setter methods.
* It provides a setter and getter methods for each added field.
* For the numpy arrays fields, the sample generated class exposes a method to create a *DataPipeline* which elaborates them.
* Fields can belong to dataset or not. If so, the field value is considered by *DatasetDiskManager* (which saves and loads it from disk), otherwise the field is ignored.
* The generated class provides a method ```get_dataset_field()``` which returns a list of string containing the name of the dataset fields. In addition, it provides other two methods ```save_field(field_name: str, path: str, file_name: str)``` and ```load_field(field_name: str, path: str, file_name: str)``` to save and load a field value respectively.
* The generated sample class instances are also **thread safe**: all methods are synchronized with respect to the fields they use. This is done using a decorator called ```synchronize_on_fields(field_names: Set[str], check_pipeline: bool)```. f*ields_name* is the set of string which contains the fields names with respect the method is synchronized and, if check_pipeline is True, an exception is raised if a specified field has an active pipeline. Also the methods created by the programmer must be synchronized: it can be done using this decorator. Also the redefined *is_positive* field can be used to synchronize methods.
* Instances of generated sample class offer other two methods to acquire and release all locks. This can be useful to perform operation to the same instance in an atomic fashion. This functionality is also implemented using context manager (*with statement*).

The *SampleGenerator* can be configured by the programmer in order to build the necessary sample class. The configuration actions are explained in the following sections. 

#### Adding of fields

You can add fields that characterize the sample class, specifying their name and type. For each field, *SampleGenerator* automatically creates:

* getter method
* setter method
* a method which returns a *DataPipeline* to elaborate this filed (only if the field is a numpy array).

A field can belong to the dataset or not. If so, the field value is considered by *DatasetDiskManager* (which saves and loads it from disk), otherwise the field is ignored.

```python
from generic_dataset.sample_generator import SampleGenerator
import generic_dataset.utilities.save_load_methods as slm
import numpy as np

GeneratedSampleClass = SampleGenerator(name='GeneratedSampleClass').add_field('field_1', field_type=int) \
    .add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .generate_sample_class()
    
generated = GeneratedSampleClass(is_positive=False)
generated_sample = GeneratedSampleClass(is_positive=False)
generated_sample.get_field_1()
generated_sample.set_field_2(np.array([]))

pipeline = generated_sample.create_pipeline_for_field_2()

# The pipeline is empty, so its result is the same as the initial value if field_2
# The get_data() method automatically sets the pipeline result in the corresponding field in the sample instance
data = pipeline.run(use_gpu=False).get_data()
```

In this case an instance of *GeneratedSampleClass* has two fields (*field_1* and *field_2*), of which only the second one belong to the dataset (*field_1* is only an instance attribute). For *field_2* (which is a numpy array), the *GeneratedSampleClass* instance provides a method to generate an elaboration pipeline. When the get_data() method is called, the pipeline result is automatically sets inside the sample instance that creates the pipeline. If a field belong to dataset, the programmer has to specify the save and load functions. This library provides a series of common function to save and load numpy arrays, OpenCV images and python dictionaries. They are defined in ```generic_dataset/utilities/save_load_methods.py```, but the programmer can defines its own functions, following these constraints:

* **save function prototype:** ```save_function(path: str, data: type) -> NoReturn``` 
* **load function prototype:** ```load_function(path: str) -> type``` 

#### Adding of predefined pipeline

The programmer can also add a predefined pipeline to elaborate a field. The pipeline result can be assigned to the same field or to another one. This can be particularly useful when a field is generated starting from another. Look at the following code.

```python
from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.sample_generator import SampleGenerator
import numpy as np

pipeline_rgb_to_gbr = DataPipeline().add_operation(lambda data, engine: (data[:, :, [2, 1, 0]], engine))

GeneratedSample = SampleGenerator(name='GeneratedSample').add_field(field_name='rgb_image') \
    .add_dataset_field(field_name='bgr_image', field_type=np.ndarray) \
    .add_custom_pipeline(method_name='create_pipeline_convert_rgb_to_bgr', elaborated_field='rgb_image', final_field='bgr_image', pipeline=pipeline_rgb_to_gbr) \
    .generate_sample_class()

rgb_image = np.array([[255, 0, 0] for _ in range(256 * 256)]).reshape((256, 256, 3))
generated_sample = GeneratedSample(is_positive=False).set_rgb_image(value=rgb_image)
generated_sample.create_pipeline_convert_rgb_to_bgr().run(use_gpu=False).get_data()
```

In this example, a custom pipeline (which convert an image from RGB to BGR) is added to the *GeneratedSample* instance. The pipeline elaborates *rgb_image* field and assigns the result to *bgr_image* field of sample instance.

#### Adding of custom methods

*SampleGenerator* provides a mechanism to add methods to the sample generated class. The programmer can define a function and assign it to the sample instance. Remember to decorate the function using *synchronize_on_fields* to make the method thread safe.

```python
from generic_dataset.sample_generator import SampleGenerator
from generic_dataset.generic_sample import synchronize_on_fields

@synchronize_on_fields(field_names={'field_1'}, check_pipeline=False)
def field_1_is_positive(sample) -> bool:
    return sample.get_field_3() > 0
GeneratedSample = SampleGenerator(name='GeneratedSample').add_field(field_name='field_1', field_type=int) \
    .add_custom_method(method_name='field_1_is_positive', function=field_1_is_positive) \
    .generate_sample_class()
    
generated = GeneratedSample(is_positive=False).set_field_1(1)
generated.field_1_is_positive()
```

As you can see, the function  field_1_is_positive is added as instance method method to the generated sample class: this method is called *field_1_is_positive()*. The function has been decorated to make the method thread safe.

### DatasetDiskManager

*DatasetDiskManager* is responsible for storing and organizing the dataset on disk. It works using the methods provided by super-type *GenericSample*: in this way *DatasetDiskManager* can operates with all sample generated classes without any change. When it is instantiated (the constructor prototype is ```DataserDiskManager(dataset_path: str, folder_name: str, slample_class: type(GenericSample))```), it automatically creates the dataset folder hierarchy (if it still doesn't exist). *dataset_path* defines the absolute path of the dataset folder, where the main folder name is specified at its end (i.e: '/home/user/dataset_main_folder'). Inside it, another folder is created: the name is given by *folder_name* parameter. This folder divides the dataset into many split, which could specified different data categories or different moments in which the data are collected. Then, two folder are created: 


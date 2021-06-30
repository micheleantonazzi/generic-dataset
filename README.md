# Generic dataset manager
![](https://github.com/micheleantonazzi/generic-dataset/workflows/Build/badge.svg?branch=main)
[![pypi](https://img.shields.io/pypi/v/generic-dataset.svg)](https://pypi.org/project/generic-dataset/)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=coverage)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)



[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=alert_status)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=sqale_rating)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=reliability_rating)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=security_rating)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=vulnerabilities)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)

This is a **configurable framework** that generates automatically the code and the necessary classes to manage a dataset of any kind. This is possible using the **metaprogramming paradigm**. The programmer can create his own dataset manager according to his needs. In addition, it also offers useful utility to manipulate *numpy arrays*. This utility builds a pipeline (a series of operations to modify an array) which can also be easily run on **GPU** without modifying the code. For this reason, this library is particularly suitable for **image datasets** or for those datasets that massively use numpy arrays. Since that the code generated at run-time suffers from the lack of type hints and auto-completion features, stub files can be automatically created using the [stub-generator package](https://github.com/micheleantonazzi/python-stub-runtime-generator).

## Installation

'generic-dataset' can be easily installed using pip:

```bash
pip install generic-dataset
```

Otherwise, you can clone this repository in your machine and then install it:

```bash
git clone https://github.com/micheleantonazzi/generic-dataset
cd generic-dataset/
pip install -e .
```

 ### GPU support

This library can accelerate the operations performed over numpy array using *Nvidia GPU*. This is possible using [CuPy](https://cupy.dev/) framework, which offers an interface highly compatible than NumPy, but all its functionalities are executed on GPU.

**NB:** you can use this library even without a GPU: if you try to use it, an exception is raised.

To enable the GPU support, please install and configure CUDA Toolkit before installing this package: this will automatically install CuPy during the installation of *generic-dataset*. Otherwise, you can configure the CUDA environment at a later time and then install CuPy using its [installation guide](https://docs.cupy.dev/en/stable/install.html#installing-cupy).

## Library structure
This library is composed by three main entities: 

* **SampleGenerator:** it is a configurable class that generates **sample classes** according to the programmers' needs. This means that the *sample class* is not apriori defined, but it is composed by *SampleGenerator* using the *metaprogramming paradigm*. The generated *sample classes* can model a *classification* or a *regression problem*, so the sample label could be an integer, which belongs to a discrete set, or a real number.  In addition to the label, a sample is characterized by fields (containing the sample data) and the operations to manipulate them.
* **DatasetFolderManager:** this class is responsible for managing a dataset folder. It can work with any type of generated sample class.
* **DataPipeline:** this entity defines an elaboration pipeline to manipulate numpy arrays. A pipeline can be executed in CPU or GPU.

### Sample class characteristics

As mentioned before, this library is not designed for a precise dataset or for a purpose clearly defined. The core of a dataset is the sample class and this framework allows you to create your own using a few lines of code. The sample classes, built by *SampleGenerator*, are sub-type of *GenericSample*, which defines some abstract method common to all sample instances. The most important aspect of a sample is the **label**. The label set defines what kind of problem the dataset models. The generated sample class could have an integer label (for a classification problem) or a real number label (for modeling a regression problem). In addition, a sample must store the data: for this purpose, a programmer can add a variable number of fields. These fields can be considered only class attributes or they can belong to the dataset: those fields must be saved and load from disk. For numpy arrays fields, the sample generated instances provide methods that create a *DataPipeline* to elaborate them. The programmer can also add custom methods to the sample class that he created. The generated sample instances have another important feature: they are **thread-safe**. In fact, all methods are synchronized to the fields they use. This is done using a decorator called *synchronize_on_fields*. It can also be configured to raise an exception if a field has an active pipeline. Also, the methods created by the programmer must be synchronized, always using this decorator. The sample label is modeled as a generic field, so it can be use to synchronize methods. In addition, instances of generated sample class offer other two methods to acquire and release all locks. This can be useful to perform multiple operations to the same instance in an atomic fashion. This functionality is also implemented using context manager (*with statement*).

### SampleGenerator usage

Using *SampleGenerator*, the programmer you can create your own customized sample class. 

#### Model a regression or a classification task

To instantiate *SampleGenerator*, the sample class name and the label set must be specified. If the label set is empty, the label assigned to the generated sample class is a real number (to model a regression problem), otherwise the label is an integer (for classification tasks). 

```python
from generic_dataset.sample_generator import SampleGenerator

SampleClassRegression = SampleGenerator(name='SampleClassRegression', label_set=set()).generate_sample_class()
sample = SampleClassRegression(label=1.1)
sample.get_label() == 1.1

SampleClassClassification= SampleGenerator(name='SampleClassClassification', label_set={-1, +1}).generate_sample_class()
sample = SampleClassClassification(label=1)
sample.get_label() == 1
```

#### How to add fields

The programmer can also add fields that characterize the sample class, specifying their name and type. For each field, *SampleGenerator* automatically creates the getter and setter methods and a function that returns a *DataPipeline* to elaborate this field (only if the field is a numpy array). A field can belong to the dataset or not. If so, the field value is considered by *DatasetFolderManager* (which saves and loads it from disk), otherwise the field is ignored.

```python
from generic_dataset.sample_generator import SampleGenerator
import generic_dataset.utilities.save_load_methods as slm
import numpy as np

GeneratedSampleClass = SampleGenerator(name='GeneratedSampleClass', label_set={0, 1}).add_field('field_1', field_type=int) \
    .add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .generate_sample_class()
    
generated = GeneratedSampleClass(label=0)
generated_sample = GeneratedSampleClass(label=0)
generated_sample.get_field_1()
generated_sample.set_field_2(np.array([]))

pipeline = generated_sample.create_pipeline_for_field_2()

# The pipeline is empty, so its result is the same as the initial value if field_2
# The get_data() method automatically sets the pipeline result in the corresponding field in the sample instance
data = pipeline.run(use_gpu=False).get_data()
```

In this case, an instance of *GeneratedSampleClass* has two fields (*field_1* and *field_2*), of which only the second one belongs to the dataset (*field_1* is only an instance attribute). For *field_2* (which is a numpy array), the *GeneratedSampleClass* instance provides a method to generate an elaboration pipeline. When the get_data() method is called, the pipeline result is automatically set to the sample instance that creates the pipeline. If a field belongs to the dataset, the programmer has to specify the save and load functions. This library provides a series of common functions to save and load numpy arrays, OpenCV images and python dictionaries. They are defined in ```generic_dataset/utilities/save_load_methods.py```, but the programmer can define its own functions, following these constraints:

* **save function prototype:** ```save_function(path: str, data: type) -> NoReturn``` 
* **load function prototype:** ```load_function(path: str) -> type``` 

#### Adding of predefined pipeline

The programmer can also add a predefined pipeline to elaborate a field. The pipeline result can be assigned to the same field or to another one. This can be particularly useful when a field is generated starting from another. Look at the following code.

```python
from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.sample_generator import SampleGenerator
import numpy as np

pipeline_rgb_to_gbr = DataPipeline().add_operation(lambda data, engine: (data[:, :, [2, 1, 0]], engine))

GeneratedSample = SampleGenerator(name='GeneratedSample', label_set=set()).add_field(field_name='rgb_image') \
    .add_dataset_field(field_name='bgr_image', field_type=np.ndarray) \
    .add_custom_pipeline(method_name='create_pipeline_convert_rgb_to_bgr', elaborated_field='rgb_image', final_field='bgr_image', pipeline=pipeline_rgb_to_gbr) \
    .generate_sample_class()

rgb_image = np.array([[255, 0, 0] for _ in range(256 * 256)]).reshape((256, 256, 3))
generated_sample = GeneratedSample(label=1.1).set_rgb_image(value=rgb_image)
generated_sample.create_pipeline_convert_rgb_to_bgr().run(use_gpu=False).get_data()
```

In this example, a custom pipeline (which convert an image from RGB to BGR) is added to the *GeneratedSample* instance. The pipeline elaborates *rgb_image* field and assigns the result to *bgr_image* field of sample instance.

#### How to add custom methods

*SampleGenerator* provides a mechanism to add methods to the sample generated class. The programmer can define a function and assign it to the sample instance. Remember to decorate the function using *synchronize_on_fields* to make the method thread-safe.

```python
from generic_dataset.sample_generator import SampleGenerator
from generic_dataset.generic_sample import synchronize_on_fields

@synchronize_on_fields(field_names={'field_1'}, check_pipeline=False)
def field_1_is_positive(sample) -> bool:
    return sample.get_field_3() > 0
GeneratedSample = SampleGenerator(name='GeneratedSample', label_set=set()).add_field(field_name='field_1', field_type=int) \
    .add_custom_method(method_name='field_1_is_positive', function=field_1_is_positive) \
    .generate_sample_class()
    
generated = GeneratedSample(is_positive=False).set_field_1(1)
generated.field_1_is_positive()
```

As you can see, the function  *field_1_is_positive* is added as an instance method to the generated sample class: this method is called *field_1_is_positive()*. The function has been decorated to make the method thread-safe.

### DatasetFolderManager

*DatasetFolderManager* is responsible for storing and organizing the dataset on disk. It works using the methods provided by the super-type *GenericSample*. In this way, *DatasetFolderManager* can operate with all sample generated classes without any change. When it is instantiated, it automatically creates the dataset folder hierarchy (if it still doesn't exist). This hierarchy is organized as follows: inside the dataset main folder, another directory is created. It divides the dataset into many split, which could specify different data categories or different moments in which the data are collected. Then, if a classification problem is modeled, a folder is created for each value in the label set, so the samples are divided according to their label. Otherwise, in a regression task, the samples are saved altogether and the label is saved as a dataset field. Finally, samples are saved grouping their fields in the same directory. Inside these folders (one for each field), the files containing the field values are named as follow: ```{field_name}_{relative_count}_({absolute_count})```, where *relative count* is the sample count depending on its label while *absolute count* is the sample total count. In the case of regression task, these numbers are equal because the samples are not divided according to the label value. The final folder hierarchy is:

```
dataset_main_folder (dir)
	- folder_classification (dir)
		- 0 (dir)
			- field_1 (dir)
				- field_1_0_(0) (file)
				- field_1_1_(2) (file)
			- field_2 (dir)
				- field_2_0_(0) (file)
				- field_2_1_(2) (file)
		- 1 (dir)
			- field_1 (dir)
				- field_1_0_(1) (file)
			- field_2 (dir)
				- field_2_0_(1) (file)
	- regression_folder (dir)
		- field_1 (dir)
			- field_1_0_(0) (file)
			- field_1_1_(1) (file)
		- field_2 (dir)
			- field_2_0_(0) (file)
			- field_2_1_(1) (file)
```

To save and load file samples, you can use the method provided by *DatasetFolderManager*.

```python
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.sample_generator import SampleGenerator
import generic_dataset.utilities.save_load_methods as slm
import numpy as np

GeneratedSampleClass = SampleGenerator(name='GeneratedSampleClass', label_set={0, 1}).add_field('field_1',
                                                                                                field_type=int)
.add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array,
                   load_function=slm.load_compressed_numpy_array)
.generate_sample_class()

database = DatasetFolderManager(dataset_path='dataset_path', folder_name='folder_classification',
                                sample_class=GeneratedSampleClassification)

sample = GeneratedSampleClass(label=0).set_field_1(np.array([0 for _ in range(1000)]))

# Save sample
database.save_sample(sample, use_thread=False)

# Load sample
for (label, relative_count) in database.get_samples_information():
    sample = database.load_sample_using_relative_count(label=label, relative_count=relative_count, use_thread=False)
```

Using large datasets, the folder's metadata calculation can be an extremely long process. To solve this issue, the folder metadata can be saved to disk: they are automatically loaded from file when a new instance of *DatasetFolderManager* is created.

```python
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.sample_generator import SampleGenerator
import generic_dataset.utilities.save_load_methods as slm
import numpy as np

GeneratedSampleClass = SampleGenerator(name='GeneratedSampleClass', label_set={0, 1}).add_field('field_1',
                                                                                                field_type=int)
.add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array,
                   load_function=slm.load_compressed_numpy_array)
.generate_sample_class()

# The folder metadata are calulcated on the fly
database = DatasetFolderManager(dataset_path='dataset_path', folder_name='folder_classification',
                                sample_class=GeneratedSampleClassification)

sample = GeneratedSampleClass(label=0).set_field_1(np.array([0 for _ in range(1000)]))

# Save sample
database.save_sample(sample, use_thread=False)

# Save folder metadata
database.save_metadata()

# The folder metadata are loaded from file
database = DatasetFolderManager(dataset_path='dataset_path', folder_name='folder_classification',
                                sample_class=GeneratedSampleClassification)
```

### DataPipeline

*DataPipeline* implements a mechanism to elaborate numpy arrays. As suggested by its name, this class creates an elaboration pipeline to modify a numpy array. A pipeline consists of a series of operations performed iteratively and it can be executed using both CPU and GPU. To do this, *DataPipeline* uses [CuPy](https://cupy.dev/) framework, which offers an interface highly compatible than NumPy, but all its functionalities are executed on GPU. This means you can write agnostic code: the pipeline can run in GPU or CPU without modifying the code, simply by replacing the engine (NumPy or CuPy). A pipeline operation consists of a function that accepts the data to modify and the used engine and returns both. This function can be simply added to a pipeline with a dedicated method. A pipeline is executed using the *run(use_gpu: bool)* method. If the method parameter is True, the pipeline is run on GPU and this method is asynchronous. This means that the pipeline is independently executed on the external device and the CPU can continue to run its operations. To synchronize them (CPU and GPU), use the method *get_data()*: it returns the pipeline result blocking the current thread until the elaboration is finished. In addition, it is possible to add a particular function called *end_function*. It is executed as the last step, when *get_data()* method is called. It allows the programmer to perform actions using the elaborated data. Its prototype is ```end_funtion(data: numpy.ndarray) -> np.ndarray```.

```python
from generic_dataset.data_pipeline import DataPipeline
import numpy as np


run_pipeline_on_gpu = False

red_image = np.array([[255, 0, 0] for _ in range(256 * 256)]).reshape((256, 256, 3))

pipeline_rgb_to_grayscale = DataPipeline() \
    .set_data(data=red_image) \
    .set_end_function(f=lambda d: d) \
    .add_operation(lambda data, engine: (engine.mean(data, axis=2), engine))

# The run method is async only if the pipeline is executed on gpu
grayscale_image = pipeline_rgb_to_grayscale.run(use_gpu=run_pipeline_on_gpu).get_data()


pipeline_rgb_to_bgr = DataPipeline() \
    .set_data(data=red_image) \
    .set_end_function(lambda d: d) \
    .add_operation(lambda data, engine: (data[..., [2, 1, 0]], engine))

bgr_image = pipeline_rgb_to_bgr.run(use_gpu=run_pipeline_on_gpu).get_data()
```




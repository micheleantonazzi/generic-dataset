import os
import shutil

import numpy as np
import pytest

from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.dataset_folder_manager import DatasetFolderManager, LabelNotFoundException
from generic_dataset.dataset_manager import DatasetManager
from generic_dataset.sample_generator import SampleGenerator
import generic_dataset.utilities.save_load_methods as slm


pipeline_field_1_2 = DataPipeline().add_operation(lambda data, engine: (engine.array([1 for i in range(10000)]), engine))
GeneratedSampleClassification = SampleGenerator(name='GeneratedSample', label_set={0, 1}) \
    .add_dataset_field(field_name='field_1', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_field(field_name='field_3', field_type=np.ndarray) \
    .add_custom_pipeline(method_name='pipeline_field_1_2', elaborated_field='field_1', final_field='field_2', pipeline=pipeline_field_1_2) \
    .generate_sample_class()

GeneratedSampleRegression = SampleGenerator(name='GeneratedSample', label_set=set()) \
    .add_dataset_field(field_name='field_1', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_dataset_field(field_name='field_2', field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=slm.load_compressed_numpy_array) \
    .add_field(field_name='field_3', field_type=np.ndarray) \
    .add_custom_pipeline(method_name='pipeline_field_1_2', elaborated_field='field_1', final_field='field_2', pipeline=pipeline_field_1_2) \
    .generate_sample_class()

path_classification = os.path.join(os.path.dirname(__file__), 'dataset_classification_folder')
path_regression = os.path.join(os.path.dirname(__file__), 'dataset_regression_folder')

shutil.rmtree(path_classification, ignore_errors=True)
shutil.rmtree(path_regression, ignore_errors=True)

# Creates classification dataset
dataset_folder_1 = DatasetFolderManager(dataset_path=path_classification, folder_name='folder_classification_1', sample_class=GeneratedSampleClassification)
dataset_folder_2 = DatasetFolderManager(dataset_path=path_classification, folder_name='folder_classification_2', sample_class=GeneratedSampleClassification)

sample = GeneratedSampleClassification(label=0).set_field_1(np.array([0.1 for _ in range(10000)])).set_field_2(np.array([0.2 for _ in range(10000)]))
dataset_folder_1.save_sample(sample, False)

sample = GeneratedSampleClassification(label=1).set_field_1(np.array([1.1 for _ in range(10000)])).set_field_2(np.array([1.2 for _ in range(10000)]))
dataset_folder_1.save_sample(sample, True).result()

sample = GeneratedSampleClassification(label=0).set_field_1(np.array([2.1 for _ in range(10000)])).set_field_2(np.array([2.2 for _ in range(10000)]))
dataset_folder_1.save_sample(sample, True).result()

sample = GeneratedSampleClassification(label=1).set_field_1(np.array([3.1 for _ in range(10000)])).set_field_2(np.array([3.2 for _ in range(10000)]))
dataset_folder_2.save_sample(sample, True).result()

sample = GeneratedSampleClassification(label=0).set_field_1(np.array([4.1 for _ in range(10000)])).set_field_2(np.array([4.2 for _ in range(10000)]))
dataset_folder_2.save_sample(sample, True).result()


# Creates dataset for classification
dataset_folder_1 = DatasetFolderManager(dataset_path=path_regression, folder_name='folder_regression_1', sample_class=GeneratedSampleRegression)
dataset_folder_2 = DatasetFolderManager(dataset_path=path_regression, folder_name='folder_regression_2', sample_class=GeneratedSampleRegression)

sample = GeneratedSampleRegression(label=.0).set_field_1(np.array([0.1 for _ in range(10000)])).set_field_2(np.array([0.2 for _ in range(10000)]))
dataset_folder_1.save_sample(sample, False)

sample = GeneratedSampleRegression(label=1.0).set_field_1(np.array([1.1 for _ in range(10000)])).set_field_2(np.array([1.2 for _ in range(10000)]))
dataset_folder_1.save_sample(sample, True).result()

sample = GeneratedSampleRegression(label=.0).set_field_1(np.array([2.1 for _ in range(10000)])).set_field_2(np.array([2.2 for _ in range(10000)]))
dataset_folder_1.save_sample(sample, True).result()

sample = GeneratedSampleRegression(label=1.0).set_field_1(np.array([3.1 for _ in range(10000)])).set_field_2(np.array([3.2 for _ in range(10000)]))
dataset_folder_2.save_sample(sample, True).result()

sample = GeneratedSampleRegression(label=.0).set_field_1(np.array([4.1 for _ in range(10000)])).set_field_2(np.array([4.2 for _ in range(10000)]))
dataset_folder_2.save_sample(sample, True).result()


def test_get_folders():
    dataset = DatasetManager(dataset_path=path_classification, sample_class=GeneratedSampleClassification, max_treads=8)
    assert dataset.get_folder_names().sort() == ['folder_classification_1', 'folder_classification_2'].sort()

    dataset = DatasetManager(dataset_path=path_regression, sample_class=GeneratedSampleRegression, max_treads=8)
    assert dataset.get_folder_names().sort() == ['folder_regression_1', 'folder_regression_2'].sort()


def test_save_metadata():
    dataset_classification = DatasetManager(dataset_path=path_classification, sample_class=GeneratedSampleClassification, max_treads=8)
    dataset_regression = DatasetManager(dataset_path=path_regression, sample_class=GeneratedSampleRegression, max_treads=8)

    dataset_regression.save_metadata()
    dataset_classification.save_metadata()

    assert os.path.exists(os.path.join(path_classification, 'folder_classification_1', 'metadata.dat')) and os.path.exists(os.path.join(path_classification, 'folder_classification_2', 'metadata.dat'))
    assert os.path.exists(os.path.join(path_regression, 'folder_regression_1', 'metadata.dat')) and os.path.exists(os.path.join(path_regression, 'folder_regression_2', 'metadata.dat'))


def test_sample_count():
    dataset_classification = DatasetManager(dataset_path=path_classification, sample_class=GeneratedSampleClassification, max_treads=8)
    dataset_regression = DatasetManager(dataset_path=path_regression, sample_class=GeneratedSampleRegression, max_treads=8)

    assert dataset_classification.get_sample_count() == {0: 3, 1: 2}
    assert dataset_regression.get_sample_count() == 5
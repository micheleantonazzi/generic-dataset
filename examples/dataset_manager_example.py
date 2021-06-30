import os

# Run dataset_folder_example_classification before run this code
from examples.generated_sample_classification import GeneratedSampleClassification
from generic_dataset.dataset_manager import DatasetManager

dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_folder_classification')

dataset = DatasetManager(dataset_path=dataset_path, sample_class=GeneratedSampleClassification, max_treads=8)

print('The folder inside the dataset are: ' + str(dataset.get_folder_names()))
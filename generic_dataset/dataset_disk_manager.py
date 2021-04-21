import os
from abc import ABCMeta

from generic_dataset.generic_sample import GenericSample


class DatasetDiskManager:
    """
    This class manages storage and loading of the dataset from disk.
    It automatically creates the dataset saving directory according to the fields of the generated sample class.
    DatasetDiskManager can automatically handle any generated sample class created using SampleGenerator.
    The dataset directory is organized as follow:
        - dataset (dir)
            - folder_1 (dir)
                - positive_samples (dir)
                    - sample_field_1 (dir)
                        positive_filed_1_sample_1 (file)
                        positive_field_1_sample_2 (file)
                    - sample_field_2 (dir)
                        positive_filed_2_sample_1 (file)
                        positive_field_2_sample_2 (file)
                - negative_samples (dir)
                    - sample_field_1 (dir)
                        .... (files)
                    - sample_field_2 (dir)
                        .... (files)
            - folder_2 (dir)


    At the top level, the samples are divided into several "folders" (they can represent different procedures to acquire the data).
    Subsequently, samples are divided into positive and negative. Finally, the sample fields are grouped according to their type
    in dedicated directories. Inside these folders, the data are ordered with respect to the sample they belong to.
    """
    POSITIVE_DATA_FOLDER = 'positive_samples'
    NEGATIVE_DATA_FOLDER = 'negative_samples'

    def __init__(self, dataset_path: str, folder_name: str, sample: GenericSample):
        """
        Instantiates a new instance of DatasetDiskManager.
        This constructor automatically creates the directory tree in which the samples are saved and loaded.
        :raise FileNotFoundError: if the dataset path does not exists
        :param dataset_path: the absolute path where to create the dataset root folder. This path incorporates the dataset root folder name: path/to/dataset/dataset_root_folder_name
        :type dataset_path: str
        :param folder_name: the folder name
        :type folder_name: str
        :param sample_class: the sample class to save and load from disk
        """
        self._dataset_path = dataset_path
        self._folder_name = folder_name
        self._sample = sample

        self._set_up_folders()

    def _set_up_folders(self):
        if not os.path.exists(os.path.dirname(self._dataset_path)):
            raise FileNotFoundError('The dataset path does not exists! \n The wrong path is ' + os.path.dirname(self._dataset_path))

        if not os.path.exists(self._dataset_path):
            os.mkdir(self._dataset_path)

        folder_dataset_path = os.path.join(self._dataset_path, self._folder_name)
        if not os.path.exists(folder_dataset_path):
            os.mkdir(folder_dataset_path)

        positive_samples_path = os.path.join(folder_dataset_path, DatasetDiskManager.POSITIVE_DATA_FOLDER)
        if not os.path.exists(positive_samples_path):
            os.mkdir(positive_samples_path)

        negative_samples_path = os.path.join(folder_dataset_path, DatasetDiskManager.NEGATIVE_DATA_FOLDER)
        if not os.path.exists(negative_samples_path):
            os.mkdir(negative_samples_path)

        dataset_fields = self._sample.get_dataset_fields()
        for folder in [positive_samples_path, negative_samples_path]:
            for field in dataset_fields:
                field_path = os.path.join(folder, field)
                if not os.path.exists(field_path):
                    os.mkdir(field_path)

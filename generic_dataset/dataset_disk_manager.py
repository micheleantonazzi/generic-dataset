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
                        positive_filed_1_sample_1_(1) (file)
                        positive_field_1_sample_4_(2) (file)
                    - sample_field_2 (dir)
                        positive_filed_2_sample_2_(1) (file)
                        positive_field_2_sample_3_(1) (file)
                - negative_samples (dir)
                    - sample_field_1 (dir)
                        .... (files)
                    - sample_field_2 (dir)
                        .... (files)
            - folder_2 (dir)


    At the top level, the samples are divided into several "folders" (they can represent different procedures to acquire the data or different data categories).
    Subsequently, samples are divided into positive and negative. Finally, the sample fields are grouped according to their type
    in dedicated directories. Inside these folders, the data are ordered with respect to the sample they belong to.
    The files are called as follow: {p}_{field_name}_sample_{c}_(n) where:
        - p = 'positive' or 'negative'
        - c = the sample progressive count
        - n = the progressive count of the sample inside its category (positive or negative)
    """
    _POSITIVE_DATA_FOLDER = 'positive_samples'
    _NEGATIVE_DATA_FOLDER = 'negative_samples'

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

        self._negative_count, self._positive_count = self._count_samples()

    def get_negative_samples_count(self) -> int:
        return self._negative_count

    def get_positive_samples_count(self) -> int:
        return self._positive_count

    def _count_samples(self):
        field = list(self._sample.get_dataset_fields())[0]
        negative_bgr_path = os.path.join(self._dataset_path, self._folder_name, DatasetDiskManager._NEGATIVE_DATA_FOLDER, field)

        count_negatives = len([name for name in os.listdir(negative_bgr_path) if os.path.isfile(os.path.join(negative_bgr_path, name))])

        positive_bgr_path = os.path.join(self._dataset_path, self._folder_name, DatasetDiskManager._POSITIVE_DATA_FOLDER, field)
        count_positives = len([name for name in os.listdir(positive_bgr_path) if os.path.isfile(os.path.join(positive_bgr_path, name))])
        return count_negatives, count_positives

    def _set_up_folders(self):
        if not os.path.exists(os.path.dirname(self._dataset_path)):
            raise FileNotFoundError('The dataset path does not exists! \n The wrong path is ' + os.path.dirname(self._dataset_path))

        if not os.path.exists(self._dataset_path):
            os.mkdir(self._dataset_path)

        folder_dataset_path = os.path.join(self._dataset_path, self._folder_name)
        if not os.path.exists(folder_dataset_path):
            os.mkdir(folder_dataset_path)

        positive_samples_path = os.path.join(folder_dataset_path, DatasetDiskManager._POSITIVE_DATA_FOLDER)
        if not os.path.exists(positive_samples_path):
            os.mkdir(positive_samples_path)

        negative_samples_path = os.path.join(folder_dataset_path, DatasetDiskManager._NEGATIVE_DATA_FOLDER)
        if not os.path.exists(negative_samples_path):
            os.mkdir(negative_samples_path)

        dataset_fields = self._sample.get_dataset_fields()
        for folder in [positive_samples_path, negative_samples_path]:
            for field in dataset_fields:
                field_path = os.path.join(folder, field)
                if not os.path.exists(field_path):
                    os.mkdir(field_path)

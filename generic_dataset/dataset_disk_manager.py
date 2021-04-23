import os
import re
import threading
from abc import ABCMeta
from concurrent.futures._base import Future
from typing import NoReturn, Union, Type, List, Tuple
from concurrent.futures import ThreadPoolExecutor

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
    The files are called as follow: {p}_{field_name}_sample_{c}_({n}) where:
        - p = 'positive' or 'negative'
        - c = the progressive count of the sample inside its category (positive or negative)
        - n = the sample absolute progressive count
    """
    _POSITIVE_DATA_FOLDER = 'positive_samples'
    _NEGATIVE_DATA_FOLDER = 'negative_samples'

    def __init__(self, dataset_path: str, folder_name: str, sample_class: type(GenericSample)):
        """
        Instantiates a new instance of DatasetDiskManager.
        This constructor automatically creates the directory tree in which the samples are saved and loaded.
        :raise FileNotFoundError: if the dataset path does not exists
        :param dataset_path: the absolute path where to create the dataset root folder. This path incorporates the dataset root folder name: path/to/dataset/dataset_root_folder_name
        :type dataset_path: str
        :param folder_name: the folder name
        :type folder_name: str
        :param sample_class: the sample class to save and load from disk
        :type sample_class: type
        """
        self._dataset_path = dataset_path
        self._folder_name = folder_name
        self._sample_class = sample_class

        self._set_up_folders()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=8)

        negative_names, positive_names = self._get_positives_negative_names()

        file_name_regexp = r'^(positive|negative)_(.+)_(\d+)_\((\d+)\)$'
        # list[i] contains the information of the positive and negative samples,
        # ordered by relative count in sample's category (positive or negative).
        # The information consist in a tuple, where tuple[i] is the absolute count of the sample
        self._positive_samples_information: List[Tuple[int]] = \
            sorted([(int(re.match(file_name_regexp, file_name).group(4)),) for file_name in positive_names], key=lambda t: t[0])
        self._negative_samples_information: List[Tuple[int]] = \
            sorted([(int(re.match(file_name_regexp, file_name).group(4)),) for file_name in negative_names], key=lambda t: t[0])

        pos_with_count = [(True, i, absolute) for i, absolute in zip(range(len(self._positive_samples_information)), self._positive_samples_information)]
        neg_with_count = [(False, i, absolute) for i, absolute in zip(range(len(self._negative_samples_information)), self._negative_samples_information)]

        # Contains the information of all samples ordered by absolute count
        # The information are contained in a tuple, where
        # tuple[0] contains a bool values with tells if the sample is positive,
        # while tuple[1] contains the sample count in its category (positive or negative)
        self._absolute_samples_information: List[Tuple[bool, int]] = [(pos, i) for pos, i, _ in sorted(pos_with_count + neg_with_count, key=lambda t: t[2])]

    def get_negative_samples_count(self) -> int:
        """
        Returns the number of negative samples in the current folder.
        :return: the number of negative samples.
        :rtype: int
        """
        with self._lock:
            return len(self._negative_samples_information)

    def get_positive_samples_count(self) -> int:
        """
        Returns the number of positive samples in the current folder.
        :return: the number of positive samples.
        :rtype: int
        """
        with self._lock:
            return len(self._positive_samples_information)

    def get_positive_samples_information(self) -> List[Tuple[int]]:
        """
        Returns the positive samples' information, sorted according to the order the positive examples
        The return value is a list with length equal to the number of examples, each cell contains a tuple, where:
            - tuple[0] = absolute sample count
        :return: a list with the positive samples information
        :rtype: List[Tuple[int]]
        """
        with self._lock:
            return self._positive_samples_information.copy()

    def get_negative_samples_information(self) -> List[Tuple[int]]:
        """
        Returns the negative samples' information, sorted according to the order the negative examples
        The return value is a list with length equal to the number of examples, each cell contains a tuple, where:
            - tuple[0]: int = absolute sample count
        :return: a list with the negative samples information
        :rtype: List[Tuple[int]]
        """
        with self._lock:
            return self._negative_samples_information.copy()

    def get_absolute_samples_information(self) -> List[Tuple[bool, int]]:
        """
        Returns the information about all sample, sorted by their absolute count.
        The return value is a list of tuples (Tuple[bool, int]), where list[i] is a tuple with the information about the i-th sample. Each tuple contains:
            - tuple[0]: bool = the sample's positiveness
            - tuple[1]: the sample count inside its category (positive or negative)
        :return: a list containing all samples information
        :rtype: List[Tuple[bool, int]]
        """
        with self._lock:
            return self._absolute_samples_information.copy()

    def save_sample(self, sample: GenericSample, use_thread: bool) -> Union[GenericSample, Future]:
        """
        Saves to disk the given sample.
        :raise TypeError: if the sample has a wrong type
        :param sample: the sample to save
        :param use_thread: if True, this method saves files in a separate thread
        :return: it use_thread is True, returns the future in which the fields is saved (the future returns the saved sample)
                returns the saved samples otherwise
        :rtype: Union[GenericSample, Future]
        """
        if not isinstance(sample, self._sample_class):
            raise TypeError('The sample type is wrong!')

        with self._lock:
            if sample.get_is_positive():
                count = len(self._positive_samples_information)
                self._positive_samples_information.append((len(self._absolute_samples_information),))
                self._absolute_samples_information.append((True, count))
            elif not sample.get_is_positive():
                count = len(self._negative_samples_information)
                self._negative_samples_information.append((len(self._absolute_samples_information),))
                self._absolute_samples_information.append((False, count))

            absolute_sample_count = len(self._absolute_samples_information) - 1

        function = self._save_or_load_sample(
            sample=sample,
            save_or_load='save',
            absolute_count=absolute_sample_count,
            relative_count=count)

        if use_thread:
            return self._pool.submit(function)
        else:
            function()

    def load_sample_using_absolute_count(self, absolute_count: int, use_thread: bool) -> Union[GenericSample, Future]:
        """
        Loads the sample with the given absolute count. The loaded sample can be positive or negative.
        :param absolute_count: the sample absolute count
        :type absolute_count: int
        :param use_thread: if True, the loading procedure is executed in a separate thread
        :type use_thread: bool
        :return: the loaded sample if the use_thread is False, otherwise the Future where the loading operation is performed
                    (the future returns the loaded sample)
        :rtype: Union[GenericSample, Future]
        """
        with self._lock:
            sample_information = self._absolute_samples_information[absolute_count]

        function = self._save_or_load_sample(
            sample=self._sample_class(is_positive=sample_information[0]),
            save_or_load='load',
            absolute_count=absolute_count,
            relative_count=sample_information[1])

        if use_thread:
            return self._pool.submit(function)
        else:
            return function()

    def load_sample_using_relative_count(self, is_positive: bool, relative_count: int, use_thread: bool) -> Union[GenericSample, Future]:
        """
        Loads the sample that has the given relative count with respect to its category (positive or negative).
        :param is_positive: the category of the sample to load
        :param relative_count: the sample's relative count
        :return: the loaded sample if use_thread is false, otherwise a future which returns the loaded sample when the procedure is completed
        :rtype: Union[GenericSample, Future]
        """
        with self._lock:
            absolute_count = self._positive_samples_information[relative_count][0] if is_positive else \
                self._negative_samples_information[relative_count][0]

        function = self._save_or_load_sample(
            sample=self._sample_class(is_positive=is_positive),
            save_or_load='load',
            absolute_count=absolute_count,
            relative_count=relative_count)

        if use_thread:
            return self._pool.submit(function)
        else:
            return function()

    def _save_or_load_sample(self, sample: GenericSample, save_or_load: str, absolute_count, relative_count):

        def f():
            with sample as sample_locked:
                fold = DatasetDiskManager._POSITIVE_DATA_FOLDER if sample_locked.get_is_positive() else DatasetDiskManager._NEGATIVE_DATA_FOLDER
                path = os.path.join(self._dataset_path, self._folder_name, fold)

                for field in sample_locked.get_dataset_fields():
                    file_name = 'positive_' if sample_locked.get_is_positive() else 'negative_'
                    file_name += field + '_' + str(relative_count) + '_('
                    file_name = file_name + str(absolute_count)
                    file_name += ')'

                    final_path = os.path.join(path, field, file_name)
                    if save_or_load == 'save':
                        sample_locked.save_field(field_name=field, path=final_path)
                    elif save_or_load == 'load':
                        sample_locked.load_field(field_name=field, path=final_path)

            return sample

        return f

    def _get_positives_negative_names(self):
        field = list(self._sample_class(is_positive=False).get_dataset_fields())[0]
        negative_path = os.path.join(self._dataset_path, self._folder_name,
                                         DatasetDiskManager._NEGATIVE_DATA_FOLDER, field)

        negative_names = [name for name in os.listdir(negative_path) if os.path.isfile(os.path.join(negative_path, name))]

        positive_path = os.path.join(self._dataset_path, self._folder_name,
                                         DatasetDiskManager._POSITIVE_DATA_FOLDER, field)

        positives_name = [name for name in os.listdir(positive_path) if os.path.isfile(os.path.join(positive_path, name))]
        return negative_names, positives_name

    def _set_up_folders(self):
        if not os.path.exists(os.path.dirname(self._dataset_path)):
            raise FileNotFoundError(
                'The dataset path does not exists! \n The wrong path is ' + os.path.dirname(self._dataset_path))

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

        dataset_fields = self._sample_class(is_positive=False).get_dataset_fields()
        for folder in [positive_samples_path, negative_samples_path]:
            for field in dataset_fields:
                field_path = os.path.join(folder, field)
                if not os.path.exists(field_path):
                    os.mkdir(field_path)

import os
import re
import threading
from concurrent.futures._base import Future
from typing import Union, Type, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

from generic_dataset.generic_sample import GenericSample
import generic_dataset.utilities.save_load_methods as slm


class LabelNotFoundException(Exception):
    pass


class DatasetDiskManager:
    """
    This class manages the persistence of the dataset.
    It automatically creates the dataset saving directory according to the label and the fields of the generated sample class.
    DatasetDiskManager can automatically handle any generated sample class, created by SampleGenerator.
    DatasetDiskManager, inside its constructor, automatically creates the folder tree where save the samples.
    In the beginning, it creates the dataset main folder.
    Inside it, the samples are divided into several directories (they can represent different procedures to acquire the data or different data categories).
    Subsequently, samples are divided according to their labels: if samples model a classification problem, a folder for each possible label is created
    and the samples are saved inside the correspondent one.
    This means that the sample label is based on the folder in which it is saved.
    Otherwise, for a regression problem, the samples are simply stored all together and the sample labels are saved within files as generic fields.
    Finally, the samples are saved grouping fields in dedicated directories (so a folder for each field is created).
    Inside these folders, the data are ordered with respect to the sample they belong to.
    The files are called as follow: {field_name}_sample_{c}_({n}) where:
        - p = 'positive' or 'negative'
        - c = the progressive count of the sample inside its category (depending on label)
        - n = the sample absolute progressive count
    Remember that for a regression task, all samples are saved in the same directory, so the relative and the absolute count of a sample are equal.
    """

    def __init__(self, dataset_path: str, folder_name: str, sample_class: Type[GenericSample], load_metadata: bool = False, max_treads: int = 4):
        """
        Instantiates a new instance of DatasetDiskManager.
        This constructor automatically creates the directory tree in which the samples are saved and loaded.
        :raise FileNotFoundError if the dataset path does not exist.
        :param dataset_path: the absolute path where to create the dataset root folder. This path incorporates the dataset root folder name: path/to/dataset/dataset_root_folder_name
        :type dataset_path: str
        :param folder_name: the folder name
        :type folder_name: str
        :param sample_class: the sample class to save and load from disk
        :type sample_class: type
        :param load_metadata: if it is true, the folder metadata are loaded from disk (if the relative file exists), otherwise the folder metadata are calculated examining the files
        :type load_metadata: bool
        :param max_treads: the max number of worker used by thread pool
        """
        self._dataset_path = dataset_path
        self._folder_name = folder_name
        self._sample_class = sample_class

        self._set_up_folders()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_treads)

        # For each label, contains a list of int, where:
        # i = sample count in its category (depending on its label)
        # list[i] = the absolute count of the i-th sample
        # (Ignored for a regression problem)
        self._label_counts: Dict[int, List[int]] = {}

        # list[i] = tuple[int, int], which are the information about the sample with i as absolute count, where
        #   - tuple[0] = contains the sample label
        #   - tuple[1] = sample count relatives to its category (label)
        # For a regression problem this list contains tuples where:
        #   - tuple[0] = 0 (label)
        #   - tuple[1] = sample absolute count
        self._absolute_samples_information: List[Tuple[int, int]] = []

        if load_metadata:
            try:
                metadata = slm.load_compressed_dictionary(os.path.join(self._dataset_path, self._folder_name, 'metadata'))
                if self._sample_class.GET_LABEL_SET():
                    self._label_counts = metadata['label_counts']
                self._absolute_samples_information = metadata['absolute_samples_information']
            except FileNotFoundError:
                self._get_sample_counts()
        else:
            self._get_sample_counts()

    def get_sample_count_in_folder(self, label: int) -> int:
        """
            Returns the number of samples with the given label in the current folder.
            If a regression problem is modeled (so the label is a real number), this methods returns the total amount of the samples.
            :raise KeyError if the label does not exist
            :param label: the label of the samples to get the total amount
            :type label: int
            :return: the number of samples.
            :rtype: int
        """
        with self._lock:
            if self._sample_class.GET_LABEL_SET():
                if label in self._sample_class.GET_LABEL_SET():
                    return len(self._label_counts[label])
                else:
                    raise LabelNotFoundException('The label {0} does not exists, the label set is {1}'.format(label, self._sample_class.GET_LABEL_SET()))
            else:
                return len(self._absolute_samples_information)

    def get_total_sample_counts(self) -> Union[int, Dict[int, int]]:
        """
        Returns the total sample counts for each label in the entire dataset.
        If the dataset models a regression problem, the total example amount is returned
        :return: for a classification problem, returns a dict containing the sample counts (dict values) for each label (dict keys).
                 for a regression problem, returns the total number of sample
        :rtype: Union[Dict[int, int], int]
        """
        with self._lock:
            if self._sample_class.GET_LABEL_SET():
                total = {label: 0 for label in self._sample_class.GET_LABEL_SET()}
                for folder in os.listdir(self._dataset_path):
                    temp = DatasetDiskManager(dataset_path=self._dataset_path, folder_name=folder, sample_class=self._sample_class)

                    for label in self._sample_class.GET_LABEL_SET():
                        total[label] += temp.get_sample_count_in_folder(label=label)

                return total
            else:
                total = 0
                for folder in os.listdir(self._dataset_path):
                    temp = DatasetDiskManager(dataset_path=self._dataset_path, folder_name=folder, sample_class=self._sample_class)
                    total += temp.get_sample_count_in_folder(label=0)

                return total

    def save_metadata(self):
        """
        This method saves to file the metadata od the current folder (
        :return:
        """
        if self._sample_class.GET_LABEL_SET():
            metadata = {'label_counts': self._label_counts, 'absolute_samples_information': self._absolute_samples_information}
        else:
            metadata = {'absolute_samples_information': self._absolute_samples_information}
        slm.save_compressed_dictionary(os.path.join(self._dataset_path, self._folder_name, 'metadata'), metadata)

    def get_samples_absolute_counts(self, label: int) -> List[int]:
        """
        Returns the absolute counts of the samples with the given label, sorted according to their order.
        The return value is a list with length equal to the number of sample with the given label.
        Each cell contains an integer: the absolute count of the i-th sample.
        If the a regression problem is modelled (so the label is a real number), this methods returns a list such that list[i] = i beacuse all samples are stored together
        :raise KeyError if the label does not exist
        :return: a list with the samples absolute counts
        :rtype: List[int]
        """
        with self._lock:
            if self._sample_class.GET_LABEL_SET():
                return self._label_counts[label].copy()
            else:
                return [count for label, count in self._absolute_samples_information]

    def get_absolute_samples_information(self) -> List[Tuple[int, int]]:
        """
        Returns the information about all sample, sorted by their absolute count.
        The return value is a list of tuples (Tuple[int, int]), where list[i] is a tuple with the information about the i-th sample. Each tuple contains:
            - tuple[0]: int = the sample's label
            - tuple[1]: the sample count inside its category (depending on its label)
        If a regression task is modelled (so the label is a real number), this methods returns a list of tuple, where list[i] =
            - tuple[0] = 0 (fake label)
            - tuple[1] = i
        :return: a list containing all samples information
        :rtype: List[Tuple[int, int]]
        """
        with self._lock:
            return self._absolute_samples_information.copy()

    def save_sample(self, sample: GenericSample, use_thread: bool) -> Union[GenericSample, Future]:
        """
        Saves to disk the given sample.
        :raise TypeError if the sample has a wrong type
        :param sample: the sample to save
        :param use_thread: if True, this method saves files in a separate thread
        :return: it use_thread is True, returns the future in which the fields is saved (the future returns the saved sample)
                returns the saved samples otherwise
        :rtype: Union[GenericSample, Future]
        """
        if not isinstance(sample, self._sample_class):
            raise TypeError('The sample type is wrong!')

        with self._lock:
            if self._sample_class.GET_LABEL_SET():
                label = sample.get_label()
                count = len(self._label_counts[label])
                absolute_count = len(self._absolute_samples_information)

                # Updated information
                self._label_counts[label].append(absolute_count)
                self._absolute_samples_information.append((label, count))
            else:
                count = absolute_count = len(self._absolute_samples_information)
                self._absolute_samples_information.append((0, count))

        function = self._save_or_load_sample(
            sample=sample,
            save_or_load='save',
            absolute_count=absolute_count,
            relative_count=count)

        if use_thread:
            return self._pool.submit(function)
        else:
            function()

    def load_sample_using_absolute_count(self, absolute_count: int, use_thread: bool) -> Union[GenericSample, Future]:
        """
        Loads the sample with the given absolute count.
        :param absolute_count: the sample absolute count
        :type absolute_count: int
        :param use_thread: if True, the loading procedure is executed in a separate thread
        :type use_thread: bool
        :return: the loaded sample if the use_thread is False, otherwise the Future where the loading operation is performed (the result() method returns the loaded sample)
        :rtype: Union[GenericSample, Future]
        """
        with self._lock:
            sample_information = self._absolute_samples_information[absolute_count]

            if self._sample_class.GET_LABEL_SET():
                sample_label = sample_information[0]
            else:
                sample_label = 0.0

        function = self._save_or_load_sample(
            sample=self._sample_class(label=sample_label),
            save_or_load='load',
            absolute_count=absolute_count,
            relative_count=sample_information[1])

        if use_thread:
            return self._pool.submit(function)
        else:
            return function()

    def load_sample_using_relative_count(self, label: int, relative_count: int, use_thread: bool) -> Union[GenericSample, Future]:
        """
        Loads the sample that has the given relative count with respect to its label.
        If a regression problem is modeled, the label is ignored and the relative count coincides with the absolute count.
        :param label: the label of the sample to load
        :param relative_count: the sample's relative count
        :return: the loaded sample if use_thread is false, otherwise a future which returns the loaded sample when the procedure is completed
        :rtype: Union[GenericSample, Future]
        """
        with self._lock:
            if self._sample_class.GET_LABEL_SET():
                absolute_count = self._label_counts[label][relative_count]
                sample_label = label
            else:
                absolute_count = relative_count
                sample_label = 0.0

        function = self._save_or_load_sample(
            sample=self._sample_class(label=sample_label),
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
                if self._sample_class.GET_LABEL_SET():
                    path = os.path.join(self._dataset_path, self._folder_name, str(sample.get_label()))
                else:
                    path = os.path.join(self._dataset_path, self._folder_name)

                for field in self._sample_class.GET_DATASET_FIELDS():
                    file_name = field + '_' + str(relative_count) + '_('
                    file_name += str(absolute_count)
                    file_name += ')'

                    if save_or_load == 'save':
                        sample_locked.save_field(field_name=field, path=os.path.join(path, field), file_name=file_name)
                    elif save_or_load == 'load':
                        sample_locked.load_field(field_name=field, path=os.path.join(path, field), file_name=file_name)

            return sample

        return f

    def _get_sample_counts(self):
        label_set = self._sample_class.GET_LABEL_SET()

        field = list(self._sample_class.GET_DATASET_FIELDS())[0]
        file_name_regexp = r'^(.+)_(\d+)_\((\d+)\).(.*)$'
        # If the label set is not empty, samples has a int label and the they are saved in different folders according to their labels
        if label_set:
            total_count: List[Tuple[int, int, int]] = []

            for label in self._sample_class.GET_LABEL_SET():
                label_path = os.path.join(self._dataset_path, self._folder_name, str(label), str(field))
                file_names = [name for name in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, name))]

                absolute_counts = [int(re.match(file_name_regexp, file_name).group(3)) for file_name in file_names]
                absolute_counts.sort()
                self._label_counts[label] = absolute_counts
                total_count += [(abs_count, label, rel_count) for rel_count, abs_count in zip(range(len(absolute_counts)), absolute_counts)]

            total_count.sort(key=lambda item: item[0])
            self._absolute_samples_information = [(label, rel_count) for abs_count, label, rel_count in total_count]
        else:
            path = os.path.join(self._dataset_path, self._folder_name, str(field))
            file_names = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
            self._absolute_samples_information = [(0, int(re.match(file_name_regexp, file_name).group(3))) for file_name in file_names]
            self._absolute_samples_information.sort(key=lambda item: item[1])

    def _set_up_folders(self):
        if not os.path.exists(os.path.dirname(self._dataset_path)):
            raise FileNotFoundError(
                'The dataset path does not exists! \n The wrong path is ' + os.path.dirname(self._dataset_path))

        if not os.path.exists(self._dataset_path):
            os.mkdir(self._dataset_path)

        folder_dataset_path = os.path.join(self._dataset_path, self._folder_name)
        if not os.path.exists(folder_dataset_path):
            os.mkdir(folder_dataset_path)

        label_set = self._sample_class.GET_LABEL_SET()
        if label_set:
            for label in label_set:
                label_path = os.path.join(folder_dataset_path, str(label))
                if not os.path.exists(label_path):
                    os.mkdir(label_path)

                dataset_fields = self._sample_class.GET_DATASET_FIELDS()
                for field in dataset_fields:
                    field_path = os.path.join(label_path, field)
                    if not os.path.exists(field_path):
                        os.mkdir(field_path)
        else:
            dataset_fields = self._sample_class.GET_DATASET_FIELDS()
            for field in dataset_fields:
                field_path = os.path.join(folder_dataset_path, field)
                if not os.path.exists(field_path):
                    os.mkdir(field_path)

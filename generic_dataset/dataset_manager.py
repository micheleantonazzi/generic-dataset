import os
from typing import Type, Dict, List, NoReturn, Union

from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.generic_sample import GenericSample


class DatasetManager:
    """
    The DatasetManager instance manages the entire dataset.
    The dataset is divided into many folders, which may contain samples acquired in different situations or conditions.
    Any folder is represented and managed by a DatasetFolderManager instance.
    DatasetManager automatically creates the dataset saving directory (if it still does not exist).
    DatasetManager can automatically handle any generated sample class, created by SampleGenerator.
    """
    def __init__(self, dataset_path: str, sample_class: Type[GenericSample], max_treads: int = 4):
        self._dataset_path = dataset_path
        self._sample_class = sample_class
        self._max_threads = max_treads

        # Create dataset folder
        if not os.path.exists(os.path.dirname(self._dataset_path)):
            raise FileNotFoundError(
                'The dataset path does not exists! \n The wrong path is ' + os.path.dirname(self._dataset_path))

        if not os.path.exists(self._dataset_path):
            os.mkdir(self._dataset_path)

        self._dataset_folder_managers: Dict[str, DatasetFolderManager] = \
            {name: DatasetFolderManager(dataset_path, name, sample_class, max_treads) for name in os.listdir(self._dataset_path)}

    def get_folder_names(self) -> List[str]:
        """
        Returns the folders' names in the dataset folder
        :return: the names of the directories
        :rtype: List[str]
        """
        return list(self._dataset_folder_managers.keys())

    def save_metadata(self) -> NoReturn:
        """
        This method saves to disk the metadata of all folders in the dataset
        :return: NoReturn
        """
        for folder_manager in self._dataset_folder_managers.values():
            folder_manager.save_metadata()

    def get_sample_count(self) -> Union[int, Dict[int, int]]:
        """
        This method returns:
            - classificiation problem: a dictionary containing the labels as keys and the relative quantity of samples for each label as values
            - regression problem: and integer value which indicates the total amount of sample in the entire dataset
        :return:
        """
        if self._sample_class.GET_LABEL_SET():
            labels = self._sample_class.GET_LABEL_SET()
            ret = {label: 0 for label in labels}
            for folder_manager in self._dataset_folder_managers.values():
                for label in labels:
                    ret[label] += folder_manager.get_sample_count(label=label)
        else:
            ret = 0
            for folder_manager in self._dataset_folder_managers.values():
                ret += folder_manager.get_sample_count(label=0)

        return ret
import os
from typing import Type, Dict, List

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

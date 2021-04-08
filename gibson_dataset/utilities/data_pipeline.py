import queue
import numpy as np
import gibson_dataset.utilities.engine_selector as eg


class DataPipeline:
    """
    This class constructs an elaboration pipeline to elaborate a single data.
    A pipeline is composed by a series of consecutive operations performed to the same data.
    A pipeline can be executed in CPU or GPU (using CuPy)
    """

    def __init__(self, data: np.array, use_gpu: bool):
        self._data = data
        self._use_gpu = use_gpu
        self._operations = queue.Queue()

        self._xp = eg.get_engine(eg.NUMPY if not use_gpu else eg.CUPY)

        # IF the pipeline is executed in gpu, transfer data to device mandatory
        if use_gpu:
            self._transfer_data_to_device()

    def _transfer_data_to_device(self) -> 'DataPipeline':
        """
        Adds to the pipeline's operations the data migration to GPU
        :return: return the pipeline
        :rtype DataPipeline
        """
        self._operations.put(lambda data: self._xp.asarray(data))

        return self

    def _get_data_from_device(self) -> 'DataPipeline':
        """
        Obtains the data from GPU
        :return: return the pipeline
        :rtype DataPipeline
        """
        self._operations.put(lambda data: self._xp.asnumpy(data))

        return self

    def run(self) -> np.array:
        """
        Runs the pipeline
        :return: None
        """
        if self._use_gpu:
            self._get_data_from_device()

        ret = self._data
        for operation in self._operations.queue:
            ret = operation(ret)

        return ret

    def convert_bgr_to_rgb(self) -> 'DataPipeline':
        """
        Adds to the pipeline's operations the conversion from BGR to RGB
        :return: the pipeline
        :rtype DataPipeline
        """
        self._operations.put(lambda data: data[..., [2, 1, 0]])

        return self

    def convert_rgb_to_bgr(self) -> 'DataPipeline':
        """
        Adds to the pipeline's operations the conversion from RGB to BGR
        :return: the pipeline
        :rtype DataPipeline
        """
        self._operations.put(lambda data: data[..., ::-1])

        return self

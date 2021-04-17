import numpy as np
from generic_dataset.utilities.data_pipeline import DataPipeline
from typing import Dict, Union, TypeVar, Callable, Any
from threading import Lock
from functools import wraps

AnotherActivePipelineException = Exception

TCallable = TypeVar('TCallable', bound=Callable[..., Any])


def synchronized_on_field(field_name: str, raise_exception: bool = True) -> Callable[[TCallable], TCallable]:
    def decorator(method: TCallable) -> TCallable:
        @wraps(method)
        def sync_method(self, *args, **kwargs):
            lock = self._locks[field_name]
            with lock:
                if raise_exception and self._pipelines[field_name] is not None:
                    raise AnotherActivePipelineException('Be careful, there is another active pipeline for {0}, please terminate it.'.format(field_name))
                return method(self, *args, **kwargs)
        return sync_method
    return decorator


class Sample:
    """
    This class represents a sample of the dataset. A sample is composed by the colored image, the relative depth data and the generated depth image.
    For manipulating each field, a pipeline can be created. There can only be one active pipeline for each field simultaneously.
    A pipeline is considered terminated when get_data() method id called.
    """
    def __init__(self, color_image: np.ndarray, depth_data: np.ndarray):
        self._color_image: np.ndarray = color_image
        self._depth_data: np.ndarray = depth_data
        self._depth_image: np.ndarray = np.empty((0, 0, 0), dtype=np.uint8)

        self._locks: Dict[str, Lock] = {'color_image': Lock(), 'depth_image': Lock(), 'depth_data': Lock()}
        self._pipelines: Dict[str, Union[DataPipeline, None]] = {'color_image': None, 'depth_image': None, 'depth_data': None}

    @synchronized_on_field(field_name='color_image', raise_exception=True)
    def create_pipeline_for_color_image(self, use_gpu: bool = False) -> DataPipeline:
        """
        Create and return a new pipeline to elaborate color_image. If there is another active pipeline, raise an AnotherActivePipelineException
        :raise AnotherActivePipelineException
        :param use_gpu: if this param is true, the pipeline is executed in GPU
        :return: a new pipeline instance
        :rtype DataPipeline
        """
        def assign(data: np.ndarray) -> np.ndarray:
            with self._locks['color_image']:
                self._color_image = data
                self._pipelines['color_image'] = None
                return data

        self._pipelines['color_image'] = DataPipeline(data=self._color_image, use_gpu=use_gpu, end_function=assign)
        return self._pipelines['color_image']

    @synchronized_on_field(field_name='depth_data', raise_exception=True)
    def create_pipeline_for_depth_data(self, use_gpu: bool = False) -> DataPipeline:
        """
        Create and return a new pipeline to elaborate depth data. If there is another active pipeline, raise an AnotherActivePipelineException.
        :raise AnotherActivePipelineException
        :param use_gpu: if this param is true, the pipeline is executed in GPU
        :type use_gpu: bool
        :return: a new pipeline instance
        :rtype DataPipeline
        """
        def assign(data: np.ndarray) -> np.ndarray:
            with self._locks['depth_data']:
                self._depth_data = data
                self._pipelines['depth_data'] = None
                return data

        self._pipelines['depth_data'] = DataPipeline(data=self._depth_data, use_gpu=use_gpu, end_function=assign)
        return self._pipelines['depth_data']

    @synchronized_on_field(field_name='depth_image', raise_exception=True)
    def create_pipeline_for_depth_image(self, use_gpu: bool = False) -> DataPipeline:
        """
        Create and return a new pipeline to elaborate depth image. If there is another active pipeline, raise an AnotherActivePipelineException.
        :raise AnotherActivePipelineException
        :param use_gpu: if this param is true, the pipeline is executed in GPU
        :type use_gpu: bool
        :return: a new pipeline instance
        :rtype DataPipeline
        """
        def assign(data: np.ndarray) -> np.ndarray:
            with self._locks['depth_image']:
                self._depth_data = data
                self._pipelines['depth_image'] = None
                return data

        self._pipelines['depth_image'] = DataPipeline(data=self._depth_image, use_gpu=use_gpu, end_function=assign)
        return self._pipelines['depth_image']

    @synchronized_on_field(field_name='depth_image', raise_exception=True)
    def create_pipeline_to_generate_depth_image(self, limit: float = 10, use_gpu: bool = False) -> DataPipeline:
        """
        Creates a pipeline to generate depth image starting from the depth data.
        The pipeline has all the necessary operations to create the depth image, you just need to run it and get the data.
        If there is another active pipeline, raise an AnotherActivePipelineException.
        :param limit: the new limit used to scale the array's values
        :type limit: float
        :param use_gpu: if this param is true, the pipeline is executed in GPU
        :type use_gpu: bool
        :return: the pipeline
        :rtype: DataPipeline
        """
        def assign(data: np.ndarray) -> np.ndarray:
            with self._locks['depth_image']:
                self._depth_image = data
                self._pipelines['depth_image'] = None
                return data

        def limit_range(data):
            data[data > limit] = limit
            return data

        self._pipelines['depth_image'] = DataPipeline(data=self._depth_data, use_gpu=use_gpu, end_function=assign)\
            .add_operation(function=limit_range).scale_values_on_new_max(new_max=255) \
            .add_operation(function=lambda data: data.astype('uint8'))

        return self._pipelines['depth_image']

    @synchronized_on_field(field_name='color_image', raise_exception=True)
    def get_color_image(self) -> np.ndarray:
        """
        Return color_image. If there is an active pipeline for this field, raises an AnotherActivePipelineException.
        :return: the colored image
        :rtype: np.ndarray
        """
        return self._color_image

    @synchronized_on_field(field_name='depth_data', raise_exception=True)
    def get_depth_data(self) -> np.ndarray:
        """
        Return depth data. If there is an active pipeline for this field, raises an AnotherActivePipelineException.
        :return: the depth data
        :rtype: np.ndarray
        """
        return self._depth_data

    @synchronized_on_field(field_name='depth_image', raise_exception=True)
    def get_depth_image(self) -> np.ndarray:
        """
        Return depth_image. If there is an active pipeline for this field, raises an AnotherActivePipelineException.
        :return: the depth image
        :rtype: np.ndarray
        """
        return self._depth_image

    @synchronized_on_field(field_name='color_image', raise_exception=False)
    def get_pipeline_color_image(self) -> DataPipeline:
        """
        Returns the color_image's pipeline
        :return: the pipeline
        :rtype: DataPipeline
        """
        return self._pipelines['color_image']

    @synchronized_on_field(field_name='depth_image', raise_exception=False)
    def get_pipeline_depth_image(self) -> DataPipeline:
        """
        Returns the depth_image's pipeline
        :return: the pipeline
        :rtype: DataPipeline
        """
        return self._pipelines['depth_image']

    @synchronized_on_field(field_name='depth_data', raise_exception=False)
    def get_pipeline_depth_data(self) -> DataPipeline:
        """
        Returns the depth_data's pipeline
        :return: the pipeline
        :rtype: DataPipeline
        """
        return self._pipelines['depth_data']

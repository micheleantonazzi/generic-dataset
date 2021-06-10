import copy
import queue
from typing import Callable

import numpy as np
import generic_dataset.utilities.engine_selector as eg


class PipelineAlreadyRunException(Exception):
    """
    This exception is raised when the user tries to change the status of a run pipeline or the pipeline is re-run.
    """
    def __init__(self):
        super(PipelineAlreadyRunException, self).__init__('The pipeline can not be modified or re-run if it has already been executed!')


class PipelineConfigurationException(Exception):
    """
    This exception is raised when a pipeline isn't correctly configured when it is run
    """
    def __init__(self):
        super(PipelineConfigurationException, self).__init__('The pipeline must be correctly configured before before is being executed')


class PipelineNotExecutedException(Exception):
    """
    This exception is raised when the pipeline has not yet been executed.
    """
    def __init__(self):
        super(PipelineNotExecutedException, self).__init__('The pipeline has not yet been executed, run it before call this method!')


class DataPipeline:
    """
    This class constructs a pipeline to elaborate a numpy.ndarray.
    A pipeline can be executed using the CPU or the GPU (using CuPy). This can be specified in the run method.
    A pipeline is composed of a series of consecutive operations performed iteratively to the same data.
    Before running a pipeline, it must be correctly configured it using the following methods:
    - set_data(): this method sets the data to elaborate
    - set_end_function(): this method sets the end-function, which can be defined by the programmer.
    - add_operation(): this method adds an operation to the pipeline.
    This configuration must be performed before calling run() method, otherwise an exception is raised.
    A pipeline cannot be re-run (so don't call run() twice)
    A series of operations can be set for each pipeline: they are iteratively executed using the initial data when run() method is called.
    The programmer can add an operation using add_operation() method.
    """

    def __init__(self):
        """
        Initializes a new pipeline.
        """
        self._data = None
        self._use_gpu = False
        self._end_function = None
        self._operations = queue.Queue()
        self._is_run = False
        self._cuda_stream = None

    def set_data(self, data: np.ndarray) -> 'DataPipeline':
        """
        Sets the data to elaborate. This method must be called before the run() function.
        :raise PipelineAlreadyRunException if the pipeline has been already run. You cannot change the data during the execution
        :param data: the data to elaborate
        :type data: numpy.ndarray
        :return: DataPipeline instance
        :rtype: DataPipeline
        """
        if self._is_run:
            raise PipelineAlreadyRunException()

        self._data = data
        return self

    def set_end_function(self, f: Callable) -> 'DataPipeline':
        """
        Sets the end-function.
        The end-function is executed when the pipeline is terminated and when the get_data() method is called.
        The end-function signature must be "f(data: numpy.ndarray) -> numpy.ndarray", where "data" is the data which have been processed.
        In this function, the programmer can put code to execute at the end of the pipeline execution.
        :raise PipelineAlreadyRunException if the pipeline has been already run (you cannot change the end-function during the pipeline execution)
        :param f: the end-function
        :type f: Callable[[numpy.ndarray], numpy.ndarray]
        :return: the pipeline
        :rtype: DataPipeline
        """
        if self._is_run:
            raise PipelineAlreadyRunException()
        self._end_function = f
        return self

    def add_operation(self, operation: Callable) -> 'DataPipeline':
        """
        Adds an operation to the pipeline.
        The operation is a function, which must have the signature "f(data, engine) -> data, engine: ...".
        "data" is the data which have been processed in the previous step, while "engine" parameter is the framework used to process the data (Numpy or Cupy).
        Remember to return both of them.
        This operation function can use indexing conventions (data[data>10]) or engine's methods (engine.around()).
        This is possible because the two frameworks are strongly compatible.
        :raise PipelineAlreadyRunException if the pipeline has been already run (you cannot add other operations)
        :param operation: the function to adds to the pipeline
        :return: the pipeline instance
        :rtype: DataPipeline
        """
        if self._is_run:
            raise PipelineAlreadyRunException()

        self._operations.put(operation)
        return self

    def run(self, use_gpu: bool) -> 'DataPipeline':
        """
        Runs the pipeline.
        Note that if the pipeline uses GPU, this is an ASYNC operation.
        To synchronize it, use the get_data() method.
        Remember to configure the pipeline before calling this method: set the data to elaborate and the end-function.
        :raise PipelineAlreadyRunException if you try to re-run the pipeline
        :raise PipelineConfigurationException if the pipeline is not correctly configured
        :param use_gpu: if the pipeline must be executed on gpu
        :type use_gpu: bool
        :return: the pipeline
        :rtype: DataPipeline
        """
        if self._is_run:
            raise PipelineAlreadyRunException()

        if self._data is None or self._end_function is None:
            raise PipelineConfigurationException()

        self._use_gpu = use_gpu
        self._is_run = True

        # Select the engine (NumPy or CuPy)
        engine = eg.get_engine(eg.NUMPY if not use_gpu else eg.CUPY)

        # IF the pipeline is executed in gpu, transfer data to device (before executing all operation)
        # and restore data from GPU (after executing the pipeline) is mandatory
        if self._use_gpu:
            self._cuda_stream = engine.cuda.Stream(null=False, non_blocking=True)
            # Transfer data from device at the end of the pipeline
            self._operations.put(lambda data, engine: (engine.asnumpy(data, stream=self._cuda_stream), engine))

        ret = self._data

        if self._use_gpu:
            with self._cuda_stream:
                ret = engine.asarray(ret)
                for operation in self._operations.queue:
                    ret, engine = operation(ret, engine)
        else:
            for operation in self._operations.queue:
                ret, engine = operation(ret, engine)

        self._data = ret

        return self

    def get_data(self) -> np.ndarray:
        """
        Returns the data after the pipeline's execution finishes and calls the end-function.
        This is a SYNC operation, so if the pipeline is running using GPU,
        the current thread is blocked until all operations in the pipeline are completed.
        :raise PipelineNotExecutedException if the pipeline has not yet been executed
        :return: the pipeline result
        :rtype: numpy.ndarray
        """
        if not self._is_run:
            raise PipelineNotExecutedException()
        if self._use_gpu:
            self._cuda_stream.synchronize()

        ret = self._end_function(self._data)

        return ret

    def set_operations(self, operations: queue.Queue) -> 'DataPipeline':
        """
        Replaces the operations queue with the input one.
        :raise PipelineAlreadyRunException if the pipeline has been already run. You cannot set the operation queue of a run pipeline
        :param operations: the queue with the operations
        :type operations: Queue
        :return: the pipeline instance
        :rtype: DataPipeline
        """
        if self._is_run:
            raise PipelineAlreadyRunException()

        self._operations = operations
        return self

    def get_operations(self) -> queue.Queue:
        """
        Returns a copy of the queue containing the pipeline operations.
        :raise PipelineAlreadyRunException if the pipeline has been already run. You cannot get the operation queue of a run pipeline
        :return: the queue with the operations
        :rtype: Queue
        """
        if self._is_run:
            raise PipelineAlreadyRunException()

        operation = copy.copy(self._operations)
        return operation
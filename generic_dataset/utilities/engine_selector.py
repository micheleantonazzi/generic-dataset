import numpy

Engine = str
NUMPY: Engine = 'numpy'
CUPY: Engine = 'cupy'


def check_cuda_support():
    """
    Print a message which tell if the current machine supports CuPy
    :return: None
    """
    try:
        import cupy as cp
        print('Success: you have CUDA Toolkit correctly installed, so you can use CuPy :)')
    except Exception as e:
        print('Something went wrong, you cant use CuPy :( \n' + str(e))


def get_engine(engine: Engine):
    """

    :raise Exception if cupy is not supported
    :param engine: the engine to return
    :type engine: Engine
    :return:
    """
    if engine == NUMPY:
        return numpy
    else:
        try:
            import cupy
            return cupy
        except Exception as e:
            raise Exception('Something went wrong, you cant use GPU :( \n' + str(e))

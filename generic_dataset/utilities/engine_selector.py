import numpy
from termcolor import colored

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
        if cp.cuda.runtime.getDeviceCount() == 0:
            print(colored('Sorry, you want to use GPU but you can\'t. You have CuPy correctly installed but not an available GPU :/', 'yellow'))
        else:
            print(colored('Success: you have CUDA Toolkit correctly installed and an available GPU, so you can use CuPy :)', 'green'))
    except Exception as e:
        print('Something went wrong, you cant use CuPy :( \n' + str(e))


def get_engine(engine: Engine):
    """
    Returns the engine (CuPy or NumPy) checking the system configuration
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
            if cupy.cuda.runtime.getDeviceCount() == 0:
                print(colored('Sorry, you want to use GPU but you can\'t. You have CuPy correctly installed but not an available GPU', 'yellow'))
                return numpy
            return cupy
        except Exception as e:
            raise Exception('Something went wrong, you cant use GPU :( \n' + str(e))

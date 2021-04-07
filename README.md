# Gibson dataset
[![pypi](https://img.shields.io/pypi/v/gibson-dataset.svg)](https://pypi.org/project/gibsion-dataset/)
![](https://travis-ci.com/micheleantonazzi/gibson-dataset.svg?branch=main)

This python library is a tool to acquire and organize data using Gibson as simulation environment.

## CUDA support
This library use Nvidia GPU to accelerate the operations over the data acquired using Gibson.
To do this, [CuPy](https://cupy.dev/) framework is used: it offers an interface highly compatible than NumPy,
but its functionalities are accelerated with CUDA.
To use CuPy's features you must have a Nvidia GPU and the CUDA Toolkit correctly installed.

**NB: you can configure gibson-dataset to work with NumPy or CuPy without any effort.
If you don't a Nvidia device correctly configured, NumPy is used as default framework.**


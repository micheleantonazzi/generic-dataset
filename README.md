# Generic dataset
![](https://github.com/micheleantonazzi/generic-dataset/workflows/Build/badge.svg?branch=main)
[![pypi](https://img.shields.io/pypi/v/generic-dataset.svg)](https://pypi.org/project/generic-dataset/)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=coverage)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)



[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=alert_status)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=sqale_rating)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=reliability_rating)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=security_rating)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)
[![](https://sonarcloud.io/api/project_badges/measure?project=micheleantonazzi_generic-dataset&metric=vulnerabilities)](https://sonarcloud.io/dashboard/index/micheleantonazzi_generic-dataset)


This python library is a tool to acquire and organize data using Gibson as simulation environment.

## CUDA support
This library use Nvidia GPU to accelerate the operations over the data acquired using Gibson.
To do this, [CuPy](https://cupy.dev/) framework is used: it offers an interface highly compatible than NumPy,
but its functionalities are accelerated with CUDA.
To use CuPy's features you must have a Nvidia GPU and the CUDA Toolkit correctly installed.

**NB: you can configure gibson-dataset to work with NumPy or CuPy without any effort.
If you don't have a Nvidia device correctly configured, NumPy is used as default framework.**


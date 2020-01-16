# pyibm

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/mesnardo/pyibm/raw/master/LICENSE)

Python Application of the Immersed Boundary Method.

## Dependencies

* Python (last tested: `3.6.10`)
* H5Py (`2.9.0`)
* Matplotlib (`3.1.1`)
* NumPy (`1.17.4`)
* Pyyaml (`5.2`)
* SciPy (`1.3.2`)
* Numba (`0.46.0`)
* CuPy (`7.0.0`), binary package for CUDA 10.1

## Installation

With Anaconda:

```shell
conda env create --name=py36-pyibm --file=environment.yaml
conda activate py36-pyibm
python setup.py install
```

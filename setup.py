#!/usr/bin/env python

import numpy
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'DEB model in C',
  ext_modules = cythonize('pydeb/cmodel.pyx'),
  include_dirs=[numpy.get_include()],
)

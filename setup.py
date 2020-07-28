#!/usr/bin/env python

import numpy
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'DEB model in C',
  ext_modules = cythonize(['pydeb/engine/optimize.pyx', 'pydeb/engine/cmodel.pyx'], language_level=3),
  include_dirs=[numpy.get_include()],
)

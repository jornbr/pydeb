#!/usr/bin/env python

import numpy
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'DEB model in C',
  ext_modules = cythonize("deb_model_eq.pyx"),
  include_dirs=[numpy.get_include()],
)

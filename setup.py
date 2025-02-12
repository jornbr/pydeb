#!/usr/bin/env python

from setuptools import setup, find_packages, Extension
import numpy

setup(
      ext_modules = [
          Extension('pydeb.engine.cmodel', ['pydeb/engine/cmodel.pyx']),
          Extension('pydeb.engine.optimize', ['pydeb/engine/optimize.pyx'])
      ],
      include_dirs=[numpy.get_include()],
)

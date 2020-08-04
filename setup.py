#!/usr/bin/env python

from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(name='pydeb',
      #use_scm_version=True,
      #setup_requires=['setuptools_scm'],
      description='Dynamic Energy Budget models in Python',
      #long_description=readme(),
      url='https://github.com/jornbr/pydeb',
      author='Jorn Bruggeman',
      author_email='jorn@bolding-bruggeman.com',
      license='GPL',
      packages=find_packages(),
      ext_modules = cythonize(['pydeb/engine/optimize.pyx', 'pydeb/engine/cmodel.pyx'], language_level=3),
      include_dirs=[numpy.get_include()],
      install_requires=['numpy'],
      classifiers=[ # Note: classifiers MUST match options on https://pypi.org/classifiers/ for PyPI submission
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Programming Language :: Python',
      ],
      entry_points={
          'console_scripts': [
                'pydeb=pydeb.__main__:main',
          ]
      },
)

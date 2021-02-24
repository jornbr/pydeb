#!/usr/bin/env python

from setuptools import setup, find_packages, Extension
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
      package_data={'pydeb.infer': ['data/*']},
      ext_modules = [
          Extension('pydeb.engine.cmodel', ['pydeb/engine/cmodel.pyx', 'pydeb/engine/optimize.pxd']),
          Extension('pydeb.engine.optimize', ['pydeb/engine/optimize.pyx'])
      ],
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

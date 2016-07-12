#!/usr/bin/env python
from setuptools import setup, find_packages, Extension

try:
  from Cython.Build import cythonize
  import numpy as np
except ImportError:
  use_cython = False
else:
  use_cython = True

setup_kwargs = dict(
    name='graphs',
    version='0.0.4',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='A library for graph-based machine learning.',
    url='http://github.com/all-umass/graphs',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    package_data={'': ['*.pyx']},
    install_requires=[
        'numpy >= 1.8',
        'scipy >= 0.14',
        'scikit-learn >= 0.15',
        'matplotlib >= 1.3.1',
        'Cython >= 0.21',
    ],
)
if use_cython:
  exts = [Extension('*', ['graphs/*/*.pyx'], include_dirs=[np.get_include()])]
  setup_kwargs['ext_modules'] = cythonize(exts)

setup(**setup_kwargs)

#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='graphs',
    version='0.0.1',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='All things graph.',
    url='http://github.com/all-umass/graphs',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy >= 1.8',
        'scipy >= 0.14',
        'scikit-learn >= 0.15',
        'matplotlib >= 1.3.1',
    ],
)

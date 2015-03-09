#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='pyroomacoustics',
      version='0.1',
      description='A simple framework for room acoustics and signal processing in Python.',
      author='Robin Scheibler, Ivan Dokmanic, Sidney Barthe',
      author_email='robin.scheibler@epfl.ch',
      url='http://lcav.epfl.ch',
      packages=['pyroomacoustics'],
	  install_requires=[
	      'numpy',
		  'scipy'],
      test_suite='nose.collector',
      tests_require=['nose']
)
